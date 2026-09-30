/*
------------------------------------------------------------------------------
Module: izhikevich_neuron

Description:
Fixed-point regular-spiking Izhikevich neuron executor. State values v and u
use signed Q(FRAC_BITS) storage. syn_current remains an integer in the
network's existing current format and is scaled internally. state_load allows
one executor to be shared by a serialized population without changing the
numerical update order.

One network timestep performs two explicit 0.5 ms Euler steps for v, followed
by one 1 ms Euler step for u:
    dv/dt = v^2 / 25 + 5v + 140 - u + I
    du/dt = a * (b*v - u)

Division truncates toward zero, matching SystemVerilog signed division. The
three constant divisors use iterative executors so no divider carry chain lies
on a single-cycle timing path.
------------------------------------------------------------------------------
*/

module izhikevich_neuron #(
    parameter int unsigned MEM_WIDTH = 24,
    parameter int unsigned CURRENT_WIDTH = 24,
    parameter int unsigned FRAC_BITS = 8,
    parameter int unsigned A_NUMERATOR = 1,
    parameter int unsigned A_DENOMINATOR = 50,
    parameter int unsigned B_NUMERATOR = 1,
    parameter int unsigned B_DENOMINATOR = 5,
    parameter integer RESET_VOLTAGE = -65,
    parameter integer RECOVERY_INCREMENT = 8,
    parameter integer SPIKE_THRESHOLD = 30
) (
    input logic clk,
    input logic rst,
    input logic clear,
    input logic state_load,
    input logic signed [MEM_WIDTH-1:0] state_voltage,
    input logic signed [MEM_WIDTH-1:0] state_recovery,
    input logic start,
    input logic signed [CURRENT_WIDTH-1:0] syn_current,
    output logic done,
    output logic spike,
    output logic signed [MEM_WIDTH-1:0] membrane,
    output logic signed [MEM_WIDTH-1:0] recovery
);
    initial begin
        if (MEM_WIDTH > 30)
            $error("izhikevich_neuron requires MEM_WIDTH <= 30 for 64-bit intermediates");
        if (FRAC_BITS >= MEM_WIDTH - 1)
            $error("izhikevich_neuron FRAC_BITS leaves no signed integer range");
        if (A_DENOMINATOR == 0 || B_DENOMINATOR == 0)
            $error("izhikevich_neuron denominators must be non-zero");
    end

    localparam int unsigned QUADRATIC_DIVISOR =
        25 * (2 ** FRAC_BITS);

    typedef enum logic [3:0] {
        IDLE,
        V_STEP_1_DIV_START,
        V_STEP_1_DIV_WAIT,
        V_STEP_1_UPDATE,
        V_STEP_2_DIV_START,
        V_STEP_2_DIV_WAIT,
        V_STEP_2_UPDATE,
        B_DIV_START,
        B_DIV_WAIT,
        U_DIV_START,
        U_DIV_WAIT,
        U_UPDATE,
        CHECK_SPIKE,
        DONE
    } state_t;

    state_t state, next_state;
    logic signed [CURRENT_WIDTH-1:0] current_reg;

    logic signed [63:0] scale_wide;
    logic signed [63:0] voltage_wide;
    logic signed [63:0] recovery_wide;
    logic signed [63:0] current_wide;
    logic signed [63:0] quadratic_dividend;
    logic signed [63:0] quadratic_quotient;
    logic quadratic_divider_done;
    logic signed [63:0] derivative_wide;
    // Only the low MEM_WIDTH bits of these wrapped sums are state.
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [63:0] voltage_sum_wide;
    logic signed [63:0] b_dividend;
    logic signed [63:0] b_quotient;
    logic b_divider_done;
    logic signed [63:0] recovery_dividend;
    logic signed [63:0] recovery_quotient;
    logic recovery_divider_done;
    logic signed [63:0] recovery_sum_wide;
    logic signed [63:0] reset_voltage_wide;
    logic signed [63:0] initial_recovery_wide;
    logic signed [63:0] threshold_wide;
    logic signed [63:0] recovery_increment_wide;
    logic signed [63:0] recovery_spike_wide;
    /* verilator lint_on UNUSEDSIGNAL */
    logic signed [63:0] a_numerator_wide;
    logic signed [63:0] b_numerator_wide;
    logic signed [63:0] b_denominator_wide;
    logic signed [MEM_WIDTH-1:0] voltage_half_next;
    logic signed [MEM_WIDTH-1:0] recovery_next;

    always_comb begin
        scale_wide = 64'sd1 <<< FRAC_BITS;
        voltage_wide = $signed({{(64-MEM_WIDTH){membrane[MEM_WIDTH-1]}}, membrane});
        recovery_wide = $signed({{(64-MEM_WIDTH){recovery[MEM_WIDTH-1]}}, recovery});
        a_numerator_wide = $signed({32'b0, A_NUMERATOR});
        b_numerator_wide = $signed({32'b0, B_NUMERATOR});
        b_denominator_wide = $signed({32'b0, B_DENOMINATOR});
        current_wide = $signed(current_reg) * scale_wide;
        quadratic_dividend = voltage_wide * voltage_wide;
        derivative_wide = quadratic_quotient + 5 * voltage_wide
            + 140 * scale_wide - recovery_wide + current_wide;
        voltage_sum_wide = voltage_wide + derivative_wide / 2;
        voltage_half_next = voltage_sum_wide[MEM_WIDTH-1:0];

        b_dividend = b_numerator_wide * voltage_wide;
        recovery_dividend =
            a_numerator_wide * (b_quotient - recovery_wide);
        recovery_sum_wide = recovery_wide + recovery_quotient;
        recovery_next = recovery_sum_wide[MEM_WIDTH-1:0];

        reset_voltage_wide = RESET_VOLTAGE * scale_wide;
        initial_recovery_wide =
            (b_numerator_wide * reset_voltage_wide) / b_denominator_wide;
        threshold_wide = SPIKE_THRESHOLD * scale_wide;
        recovery_increment_wide = RECOVERY_INCREMENT * scale_wide;
        recovery_spike_wide = recovery_wide + recovery_increment_wide;
    end

    signed_round_divider #(
        .WIDTH(64),
        .DIVISOR(QUADRATIC_DIVISOR),
        .ROUND_NEAREST(1'b0)
    ) u_quadratic_divider (
        .clk(clk),
        .rst(rst || clear),
        .start(state == V_STEP_1_DIV_START
            || state == V_STEP_2_DIV_START),
        .value(quadratic_dividend),
        .done(quadratic_divider_done),
        .quotient(quadratic_quotient)
    );

    signed_round_divider #(
        .WIDTH(64),
        .DIVISOR(B_DENOMINATOR),
        .ROUND_NEAREST(1'b0)
    ) u_b_divider (
        .clk(clk),
        .rst(rst || clear),
        .start(state == B_DIV_START),
        .value(b_dividend),
        .done(b_divider_done),
        .quotient(b_quotient)
    );

    signed_round_divider #(
        .WIDTH(64),
        .DIVISOR(A_DENOMINATOR),
        .ROUND_NEAREST(1'b0)
    ) u_recovery_divider (
        .clk(clk),
        .rst(rst || clear),
        .start(state == U_DIV_START),
        .value(recovery_dividend),
        .done(recovery_divider_done),
        .quotient(recovery_quotient)
    );

    always_ff @(posedge clk or posedge rst) begin
        if (rst || clear)
            state <= IDLE;
        else
            state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE: if (start) next_state = V_STEP_1_DIV_START;
            V_STEP_1_DIV_START: next_state = V_STEP_1_DIV_WAIT;
            V_STEP_1_DIV_WAIT:
                if (quadratic_divider_done) next_state = V_STEP_1_UPDATE;
            V_STEP_1_UPDATE: next_state = V_STEP_2_DIV_START;
            V_STEP_2_DIV_START: next_state = V_STEP_2_DIV_WAIT;
            V_STEP_2_DIV_WAIT:
                if (quadratic_divider_done) next_state = V_STEP_2_UPDATE;
            V_STEP_2_UPDATE: next_state = B_DIV_START;
            B_DIV_START: next_state = B_DIV_WAIT;
            B_DIV_WAIT: if (b_divider_done) next_state = U_DIV_START;
            U_DIV_START: next_state = U_DIV_WAIT;
            U_DIV_WAIT:
                if (recovery_divider_done) next_state = U_UPDATE;
            U_UPDATE: next_state = CHECK_SPIKE;
            CHECK_SPIKE: next_state = DONE;
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst || clear) begin
            membrane <= reset_voltage_wide[MEM_WIDTH-1:0];
            recovery <= initial_recovery_wide[MEM_WIDTH-1:0];
            current_reg <= '0;
            spike <= 1'b0;
            done <= 1'b0;
        end else begin
            spike <= 1'b0;
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (state_load) begin
                        membrane <= state_voltage;
                        recovery <= state_recovery;
                    end else if (start)
                        current_reg <= syn_current;
                end
                V_STEP_1_UPDATE, V_STEP_2_UPDATE:
                    membrane <= voltage_half_next;
                U_UPDATE: recovery <= recovery_next;
                CHECK_SPIKE: begin
                    if (voltage_wide >= threshold_wide) begin
                        membrane <= reset_voltage_wide[MEM_WIDTH-1:0];
                        recovery <= recovery_spike_wide[MEM_WIDTH-1:0];
                        spike <= 1'b1;
                    end
                end
                DONE: done <= 1'b1;
                default: begin
                end
            endcase
        end
    end
endmodule
