/*
------------------------------------------------------------------------------
Module: hodgkin_huxley_neuron

Description:
Pipelined fixed-point executor for one original squid-axon Hodgkin-Huxley
neuron. Voltage is signed 24/Q12, m/h/n are unsigned 20/Q18 and rates are
24/Q18. Six synchronous ROMs plus six arithmetic phases evaluate each of the
100 explicit-Euler substeps. state_load allows one executor to be shared by a
serialized population without changing the numerical update order.
------------------------------------------------------------------------------
*/

module hodgkin_huxley_neuron #(
    parameter int unsigned MEM_WIDTH = 24,
    parameter int unsigned CURRENT_WIDTH = 24,
    parameter LUT_DIR = "data/hh_luts/"
) (
    input logic clk,
    input logic rst,
    input logic clear,
    input logic state_load,
    input logic signed [MEM_WIDTH-1:0] state_voltage,
    input logic [19:0] state_sodium_activation,
    input logic [19:0] state_sodium_inactivation,
    input logic [19:0] state_potassium_activation,
    input logic start,
    input logic signed [CURRENT_WIDTH-1:0] syn_current,
    output logic done,
    output logic spike,
    output logic signed [MEM_WIDTH-1:0] membrane,
    output logic [19:0] sodium_activation,
    output logic [19:0] sodium_inactivation,
    output logic [19:0] potassium_activation
);
    localparam logic signed [63:0] GATE_ONE = 64'sd262144;
    localparam logic signed [63:0] VOLTAGE_MIN = -64'sd8388608;
    localparam logic signed [63:0] VOLTAGE_MAX = 64'sd8388607;
    localparam logic signed [63:0] SODIUM_REVERSAL = 64'sd471040;
    localparam logic signed [63:0] POTASSIUM_REVERSAL = -64'sd49152;
    localparam logic signed [63:0] LEAK_REVERSAL = 64'sd43471;
    localparam logic signed [63:0] SPIKE_THRESHOLD = 64'sd81920;

    initial begin
        if (MEM_WIDTH != 24)
            $error("hodgkin_huxley_neuron requires MEM_WIDTH=24");
    end

    function automatic logic signed [63:0] round_div(
        input logic signed [63:0] value,
        input logic signed [63:0] divisor
    );
        if (value >= 0)
            round_div = (value + divisor / 2) / divisor;
        else
            round_div = -((-value + divisor / 2) / divisor);
    endfunction

    /* verilator lint_off UNUSEDSIGNAL */
    function automatic logic signed [39:0] narrow40(
        input logic signed [63:0] value
    );
        narrow40 = value[39:0];
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */

    function automatic logic signed [63:0] extend40(
        input logic signed [39:0] value
    );
        extend40 = {{24{value[39]}}, value};
    endfunction

    function automatic logic signed [63:0] clamp_voltage(
        input logic signed [63:0] value
    );
        if (value < VOLTAGE_MIN)
            clamp_voltage = VOLTAGE_MIN;
        else if (value > VOLTAGE_MAX)
            clamp_voltage = VOLTAGE_MAX;
        else
            clamp_voltage = value;
    endfunction

    function automatic logic [19:0] clamp_gate(
        input logic signed [63:0] value
    );
        if (value < 0)
            clamp_gate = 20'd0;
        else if (value > GATE_ONE)
            clamp_gate = 20'd262144;
        else
            clamp_gate = value[19:0];
    endfunction

    typedef enum logic [3:0] {
        IDLE,
        LUT_READ,
        POWER_2,
        POWER_4,
        M3H_AND_K,
        POTASSIUM,
        LEAK_WAIT,
        SODIUM_PRODUCT,
        SODIUM,
        DERIVATIVE,
        DELTA_START,
        DELTA_WAIT,
        UPDATE,
        COMMIT,
        DONE_STATE
    } control_state_t;
    control_state_t control_state;

    logic [6:0] substep_count;
    logic signed [CURRENT_WIDTH-1:0] current_reg;
    logic spike_pending;
    logic [9:0] lut_address;
    logic signed [63:0] lut_address_wide;
    logic signed [63:0] voltage_wide;
    logic signed [63:0] m_wide;
    logic signed [63:0] h_wide;
    logic signed [63:0] n_wide;

    logic signed [23:0] alpha_m_reg;
    logic signed [23:0] beta_m_reg;
    logic signed [23:0] alpha_h_reg;
    logic signed [23:0] beta_h_reg;
    logic signed [23:0] alpha_n_reg;
    logic signed [23:0] beta_n_reg;
    logic signed [63:0] m2_reg;
    logic signed [63:0] m3_reg;
    logic signed [63:0] m3h_reg;
    logic signed [63:0] n2_reg;
    logic signed [63:0] n4_reg;
    logic signed [63:0] sodium_product_reg;
    logic signed [63:0] potassium_product_reg;
    logic signed [39:0] dm_reg;
    logic signed [39:0] dh_reg;
    logic signed [39:0] dn_reg;
    logic signed [39:0] sodium_current_reg;
    logic signed [39:0] potassium_current_reg;
    logic signed [39:0] leak_numerator_reg;
    logic signed [39:0] leak_current_reg;
    logic signed [39:0] voltage_derivative_reg;
    logic signed [39:0] voltage_delta_reg;
    logic signed [39:0] m_delta_reg;
    logic signed [39:0] h_delta_reg;
    logic signed [39:0] n_delta_reg;
    logic signed [39:0] voltage_next_reg;
    logic [19:0] m_next_reg;
    logic [19:0] h_next_reg;
    logic [19:0] n_next_reg;
    logic voltage_divider_done;
    logic leak_divider_done;
    logic m_divider_done;
    logic h_divider_done;
    logic n_divider_done;
    logic signed [39:0] voltage_divider_quotient;
    logic signed [39:0] leak_divider_quotient;
    logic signed [39:0] m_divider_quotient;
    logic signed [39:0] h_divider_quotient;
    logic signed [39:0] n_divider_quotient;

    logic signed [63:0] alpha_m_wide;
    logic signed [63:0] beta_m_wide;
    logic signed [63:0] alpha_h_wide;
    logic signed [63:0] beta_h_wide;
    logic signed [63:0] alpha_n_wide;
    logic signed [63:0] beta_n_wide;

    hh_rate_rom #(.INIT_FILE({LUT_DIR, "hh_alpha_m.mem"})) alpha_m_lut (
        .clk(clk), .address(lut_address), .data(alpha_m_reg)
    );
    hh_rate_rom #(.INIT_FILE({LUT_DIR, "hh_beta_m.mem"})) beta_m_lut (
        .clk(clk), .address(lut_address), .data(beta_m_reg)
    );
    hh_rate_rom #(.INIT_FILE({LUT_DIR, "hh_alpha_h.mem"})) alpha_h_lut (
        .clk(clk), .address(lut_address), .data(alpha_h_reg)
    );
    hh_rate_rom #(.INIT_FILE({LUT_DIR, "hh_beta_h.mem"})) beta_h_lut (
        .clk(clk), .address(lut_address), .data(beta_h_reg)
    );
    hh_rate_rom #(.INIT_FILE({LUT_DIR, "hh_alpha_n.mem"})) alpha_n_lut (
        .clk(clk), .address(lut_address), .data(alpha_n_reg)
    );
    hh_rate_rom #(.INIT_FILE({LUT_DIR, "hh_beta_n.mem"})) beta_n_lut (
        .clk(clk), .address(lut_address), .data(beta_n_reg)
    );

    signed_round_divider voltage_divider (
        .clk(clk), .rst(rst || clear),
        .start(control_state == DELTA_START),
        .value(voltage_derivative_reg),
        .done(voltage_divider_done), .quotient(voltage_divider_quotient)
    );
    signed_round_divider #(.DIVISOR(10)) leak_divider (
        .clk(clk), .rst(rst || clear),
        .start(control_state == POTASSIUM), .value(leak_numerator_reg),
        .done(leak_divider_done), .quotient(leak_divider_quotient)
    );
    signed_round_divider m_divider (
        .clk(clk), .rst(rst || clear),
        .start(control_state == DELTA_START), .value(dm_reg),
        .done(m_divider_done), .quotient(m_divider_quotient)
    );
    signed_round_divider h_divider (
        .clk(clk), .rst(rst || clear),
        .start(control_state == DELTA_START), .value(dh_reg),
        .done(h_divider_done), .quotient(h_divider_quotient)
    );
    signed_round_divider n_divider (
        .clk(clk), .rst(rst || clear),
        .start(control_state == DELTA_START), .value(dn_reg),
        .done(n_divider_done), .quotient(n_divider_quotient)
    );

    always_comb begin
        voltage_wide = {{40{membrane[23]}}, membrane};
        m_wide = {44'b0, sodium_activation};
        h_wide = {44'b0, sodium_inactivation};
        n_wide = {44'b0, potassium_activation};

        lut_address_wide = (voltage_wide >>> 10) + 128;
        if (lut_address_wide < 0)
            lut_address = 10'd0;
        else if (lut_address_wide >= 64'sd641)
            lut_address = 10'd640;
        else
            lut_address = lut_address_wide[9:0];

        alpha_m_wide = {{40{alpha_m_reg[23]}}, alpha_m_reg};
        beta_m_wide = {{40{beta_m_reg[23]}}, beta_m_reg};
        alpha_h_wide = {{40{alpha_h_reg[23]}}, alpha_h_reg};
        beta_h_wide = {{40{beta_h_reg[23]}}, beta_h_reg};
        alpha_n_wide = {{40{alpha_n_reg[23]}}, alpha_n_reg};
        beta_n_wide = {{40{beta_n_reg[23]}}, beta_n_reg};

    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst || clear) begin
            control_state <= IDLE;
            substep_count <= 0;
            current_reg <= 0;
            membrane <= 24'sd0;
            sodium_activation <= 20'd13876;
            sodium_inactivation <= 20'd156269;
            potassium_activation <= 20'd83277;
            spike_pending <= 1'b0;
            spike <= 1'b0;
            done <= 1'b0;
            m2_reg <= 0;
            m3_reg <= 0;
            m3h_reg <= 0;
            n2_reg <= 0;
            n4_reg <= 0;
            sodium_product_reg <= 0;
            potassium_product_reg <= 0;
            dm_reg <= 0;
            dh_reg <= 0;
            dn_reg <= 0;
            sodium_current_reg <= 0;
            potassium_current_reg <= 0;
            leak_numerator_reg <= 0;
            leak_current_reg <= 0;
            voltage_derivative_reg <= 0;
            voltage_delta_reg <= 0;
            m_delta_reg <= 0;
            h_delta_reg <= 0;
            n_delta_reg <= 0;
            voltage_next_reg <= 0;
            m_next_reg <= 0;
            h_next_reg <= 0;
            n_next_reg <= 0;
        end else begin
            spike <= 1'b0;
            done <= 1'b0;
            case (control_state)
                IDLE: begin
                    if (state_load) begin
                        membrane <= state_voltage;
                        sodium_activation <= state_sodium_activation;
                        sodium_inactivation <= state_sodium_inactivation;
                        potassium_activation <= state_potassium_activation;
                        spike_pending <= 1'b0;
                    end else if (start) begin
                        current_reg <= syn_current;
                        substep_count <= 0;
                        spike_pending <= 1'b0;
                        control_state <= LUT_READ;
                    end
                end
                LUT_READ: begin
                    control_state <= POWER_2;
                end
                POWER_2: begin
                    m2_reg <= round_div(m_wide * m_wide, GATE_ONE);
                    n2_reg <= round_div(n_wide * n_wide, GATE_ONE);
                    dm_reg <= narrow40(round_div(
                        alpha_m_wide * (GATE_ONE - m_wide), GATE_ONE
                    ) - round_div(beta_m_wide * m_wide, GATE_ONE));
                    dh_reg <= narrow40(round_div(
                        alpha_h_wide * (GATE_ONE - h_wide), GATE_ONE
                    ) - round_div(beta_h_wide * h_wide, GATE_ONE));
                    dn_reg <= narrow40(round_div(
                        alpha_n_wide * (GATE_ONE - n_wide), GATE_ONE
                    ) - round_div(beta_n_wide * n_wide, GATE_ONE));
                    control_state <= POWER_4;
                end
                POWER_4: begin
                    m3_reg <= round_div(m2_reg * m_wide, GATE_ONE);
                    n4_reg <= round_div(n2_reg * n2_reg, GATE_ONE);
                    control_state <= M3H_AND_K;
                end
                M3H_AND_K: begin
                    m3h_reg <= round_div(m3_reg * h_wide, GATE_ONE);
                    potassium_product_reg <= n4_reg *
                        (voltage_wide - POTASSIUM_REVERSAL);
                    leak_numerator_reg <= narrow40(
                        3 * (voltage_wide - LEAK_REVERSAL)
                    );
                    control_state <= POTASSIUM;
                end
                POTASSIUM: begin
                    potassium_current_reg <= narrow40(36 * round_div(
                        potassium_product_reg, GATE_ONE
                    ));
                    control_state <= SODIUM_PRODUCT;
                end
                SODIUM_PRODUCT: begin
                    sodium_product_reg <= m3h_reg *
                        (voltage_wide - SODIUM_REVERSAL);
                    control_state <= SODIUM;
                end
                SODIUM: begin
                    sodium_current_reg <= narrow40(120 * round_div(
                        sodium_product_reg, GATE_ONE
                    ));
                    control_state <= LEAK_WAIT;
                end
                LEAK_WAIT: begin
                    if (leak_divider_done) begin
                        leak_current_reg <= leak_divider_quotient;
                        control_state <= DERIVATIVE;
                    end
                end
                DERIVATIVE: begin
                    voltage_derivative_reg <= $signed(current_reg) * 4096
                        - sodium_current_reg - potassium_current_reg
                        - leak_current_reg;
                    control_state <= DELTA_START;
                end
                DELTA_START: begin
                    control_state <= DELTA_WAIT;
                end
                DELTA_WAIT: begin
                    if (voltage_divider_done && m_divider_done
                            && h_divider_done && n_divider_done) begin
                        voltage_delta_reg <= voltage_divider_quotient;
                        m_delta_reg <= m_divider_quotient;
                        h_delta_reg <= h_divider_quotient;
                        n_delta_reg <= n_divider_quotient;
                        control_state <= UPDATE;
                    end
                end
                UPDATE: begin
                    voltage_next_reg <= narrow40(clamp_voltage(
                        voltage_wide + extend40(voltage_delta_reg)
                    ));
                    m_next_reg <= clamp_gate(m_wide + extend40(m_delta_reg));
                    h_next_reg <= clamp_gate(h_wide + extend40(h_delta_reg));
                    n_next_reg <= clamp_gate(n_wide + extend40(n_delta_reg));
                    control_state <= COMMIT;
                end
                COMMIT: begin
                    membrane <= voltage_next_reg[23:0];
                    sodium_activation <= m_next_reg;
                    sodium_inactivation <= h_next_reg;
                    potassium_activation <= n_next_reg;
                    if (voltage_wide < SPIKE_THRESHOLD
                            && extend40(voltage_next_reg) >= SPIKE_THRESHOLD)
                        spike_pending <= 1'b1;
                    if (substep_count == 99) begin
                        control_state <= DONE_STATE;
                    end else begin
                        substep_count <= substep_count + 1'b1;
                        control_state <= LUT_READ;
                    end
                end
                DONE_STATE: begin
                    done <= 1'b1;
                    spike <= spike_pending;
                    control_state <= IDLE;
                end
                default: control_state <= IDLE;
            endcase
        end
    end
endmodule
