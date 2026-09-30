/*
------------------------------------------------------------------------------
Module: neuron_array

Description:
Selectable neuron population. EXECUTOR_COUNT selects the LIF population
architecture: NUM_NEURONS retains one parallel core per neuron, while 1 uses
one time-multiplexed executor and per-neuron state storage. Izhikevich and
Hodgkin-Huxley currently require EXECUTOR_COUNT=1 and use their existing shared
executor. Intermediate executor counts are reserved for the later Pareto sweep.
------------------------------------------------------------------------------
*/

module neuron_array #(
    parameter int unsigned NUM_NEURONS   = 16,
    parameter int unsigned MEM_WIDTH     = 16,
    parameter int unsigned CURRENT_WIDTH = 16,
    parameter int unsigned NEURON_MODEL  = 0,
    parameter int unsigned EXECUTOR_COUNT =
        (NEURON_MODEL == 0) ? NUM_NEURONS : 1,
    parameter THRESHOLD     = 100,
    parameter RESET_POT     = 0,
    parameter LEAK_SHIFT    = 2,
    parameter int unsigned IZH_FRAC_BITS = 8,
    parameter int unsigned IZH_A_NUMERATOR = 1,
    parameter int unsigned IZH_A_DENOMINATOR = 50,
    parameter int unsigned IZH_B_NUMERATOR = 1,
    parameter int unsigned IZH_B_DENOMINATOR = 5,
    parameter integer IZH_RESET_VOLTAGE = -65,
    parameter integer IZH_RECOVERY_INCREMENT = 8,
    parameter integer IZH_SPIKE_THRESHOLD = 30,
    parameter HH_LUT_DIR = "data/hh_luts/"
) (
    input logic clk,
    input logic rst,
    input logic clear,
    input logic update_valid,
    input logic signed [CURRENT_WIDTH-1:0] syn_current [NUM_NEURONS],
    output logic [NUM_NEURONS-1:0] spike,
    output logic signed [MEM_WIDTH-1:0] membrane [NUM_NEURONS],
    output logic done
);
    localparam int unsigned NEURON_MODEL_IZHIKEVICH = 1;
    localparam int unsigned NEURON_MODEL_HODGKIN_HUXLEY = 2;

    initial begin
        if (EXECUTOR_COUNT < 1 || EXECUTOR_COUNT > NUM_NEURONS)
            $fatal(1, "EXECUTOR_COUNT must be in [1, NUM_NEURONS]");
        if (NEURON_MODEL != 0 && EXECUTOR_COUNT != 1)
            $fatal(1, "Izhikevich and HH currently require EXECUTOR_COUNT=1");
        if (NEURON_MODEL == 0 && EXECUTOR_COUNT != 1 &&
                EXECUTOR_COUNT != NUM_NEURONS)
            $fatal(1, "LIF currently supports EXECUTOR_COUNT=1 or NUM_NEURONS");
    end

    generate
        if (NEURON_MODEL == 0 && EXECUTOR_COUNT == 1) begin : gen_lif_serial
            localparam int unsigned INDEX_WIDTH =
                (NUM_NEURONS <= 1) ? 1 : $clog2(NUM_NEURONS);
            typedef enum logic [2:0] {
                LIF_IDLE,
                LIF_LOAD,
                LIF_START,
                LIF_WAIT,
                LIF_DONE
            } lif_state_t;

            lif_state_t lif_state, lif_next_state;
            logic [INDEX_WIDTH-1:0] neuron_index;
            logic engine_done;
            logic engine_spike;
            logic engine_spike_pending;
            logic signed [MEM_WIDTH-1:0] engine_membrane;
            integer reset_index;

            assign done = (lif_state == LIF_DONE);

            always_comb begin
                lif_next_state = lif_state;
                case (lif_state)
                    LIF_IDLE: if (update_valid) lif_next_state = LIF_LOAD;
                    LIF_LOAD: lif_next_state = LIF_START;
                    LIF_START: lif_next_state = LIF_WAIT;
                    LIF_WAIT: begin
                        if (engine_done) begin
                            if (neuron_index == NUM_NEURONS - 1)
                                lif_next_state = LIF_DONE;
                            else
                                lif_next_state = LIF_LOAD;
                        end
                    end
                    LIF_DONE: lif_next_state = LIF_IDLE;
                    default: lif_next_state = LIF_IDLE;
                endcase
            end

            always_ff @(posedge clk or posedge rst) begin
                if (rst || clear) begin
                    lif_state <= LIF_IDLE;
                    neuron_index <= '0;
                    spike <= '0;
                    engine_spike_pending <= 1'b0;
                    for (reset_index = 0; reset_index < NUM_NEURONS;
                            reset_index = reset_index + 1)
                        membrane[reset_index] <= RESET_POT;
                end else begin
                    lif_state <= lif_next_state;
                    if (lif_state == LIF_IDLE && update_valid) begin
                        neuron_index <= '0;
                        spike <= '0;
                    end
                    if (lif_state == LIF_LOAD)
                        engine_spike_pending <= 1'b0;
                    else if (engine_spike)
                        engine_spike_pending <= 1'b1;
                    if (lif_state == LIF_WAIT && engine_done) begin
                        membrane[neuron_index] <= engine_membrane;
                        spike[neuron_index] <=
                            engine_spike_pending || engine_spike;
                        if (neuron_index != NUM_NEURONS - 1)
                            neuron_index <= neuron_index + 1'b1;
                    end
                end
            end

            lif_neuron #(
                .MEM_WIDTH(MEM_WIDTH),
                .CURRENT_WIDTH(CURRENT_WIDTH),
                .THRESHOLD(THRESHOLD),
                .RESET_POT(RESET_POT),
                .LEAK_SHIFT(LEAK_SHIFT)
            ) u_shared_lif (
                .clk(clk),
                .rst(rst),
                .clear(clear),
                .state_load(lif_state == LIF_LOAD),
                .state_membrane(membrane[neuron_index]),
                .start(lif_state == LIF_START),
                .syn_current(syn_current[neuron_index]),
                .done(engine_done),
                .spike(engine_spike),
                .membrane(engine_membrane)
            );
        end else if (NEURON_MODEL == NEURON_MODEL_IZHIKEVICH) begin : gen_izh_serial
            localparam int unsigned INDEX_WIDTH =
                (NUM_NEURONS <= 1) ? 1 : $clog2(NUM_NEURONS);
            localparam logic signed [63:0] IZH_B_NUMERATOR_WIDE =
                IZH_B_NUMERATOR;
            localparam logic signed [63:0] IZH_B_DENOMINATOR_WIDE =
                IZH_B_DENOMINATOR;
            localparam logic signed [63:0] INITIAL_VOLTAGE_WIDE =
                IZH_RESET_VOLTAGE * (64'sd1 <<< IZH_FRAC_BITS);
            localparam logic signed [63:0] INITIAL_RECOVERY_WIDE =
                (IZH_B_NUMERATOR_WIDE * INITIAL_VOLTAGE_WIDE)
                / IZH_B_DENOMINATOR_WIDE;
            typedef enum logic [2:0] {
                IZH_IDLE,
                IZH_LOAD,
                IZH_START,
                IZH_WAIT,
                IZH_DONE
            } izh_state_t;

            izh_state_t izh_state, izh_next_state;
            logic [INDEX_WIDTH-1:0] neuron_index;
            logic signed [MEM_WIDTH-1:0] recovery_state [NUM_NEURONS];
            logic engine_done;
            logic engine_spike;
            logic engine_spike_pending;
            logic signed [MEM_WIDTH-1:0] engine_voltage;
            logic signed [MEM_WIDTH-1:0] engine_recovery;
            integer reset_index;

            assign done = (izh_state == IZH_DONE);

            always_comb begin
                izh_next_state = izh_state;
                case (izh_state)
                    IZH_IDLE: if (update_valid) izh_next_state = IZH_LOAD;
                    IZH_LOAD: izh_next_state = IZH_START;
                    IZH_START: izh_next_state = IZH_WAIT;
                    IZH_WAIT: begin
                        if (engine_done) begin
                            if (neuron_index == NUM_NEURONS - 1)
                                izh_next_state = IZH_DONE;
                            else
                                izh_next_state = IZH_LOAD;
                        end
                    end
                    IZH_DONE: izh_next_state = IZH_IDLE;
                    default: izh_next_state = IZH_IDLE;
                endcase
            end

            always_ff @(posedge clk or posedge rst) begin
                if (rst || clear) begin
                    izh_state <= IZH_IDLE;
                    neuron_index <= '0;
                    spike <= '0;
                    engine_spike_pending <= 1'b0;
                    for (reset_index = 0; reset_index < NUM_NEURONS;
                            reset_index = reset_index + 1) begin
                        membrane[reset_index] <=
                            INITIAL_VOLTAGE_WIDE[MEM_WIDTH-1:0];
                        recovery_state[reset_index] <=
                            INITIAL_RECOVERY_WIDE[MEM_WIDTH-1:0];
                    end
                end else begin
                    izh_state <= izh_next_state;
                    if (izh_state == IZH_IDLE && update_valid) begin
                        neuron_index <= '0;
                        spike <= '0;
                    end
                    if (izh_state == IZH_LOAD)
                        engine_spike_pending <= 1'b0;
                    else if (engine_spike)
                        engine_spike_pending <= 1'b1;
                    if (izh_state == IZH_WAIT && engine_done) begin
                        membrane[neuron_index] <= engine_voltage;
                        recovery_state[neuron_index] <= engine_recovery;
                        spike[neuron_index] <=
                            engine_spike_pending || engine_spike;
                        if (neuron_index != NUM_NEURONS - 1)
                            neuron_index <= neuron_index + 1'b1;
                    end
                end
            end

            izhikevich_neuron #(
                .MEM_WIDTH(MEM_WIDTH),
                .CURRENT_WIDTH(CURRENT_WIDTH),
                .FRAC_BITS(IZH_FRAC_BITS),
                .A_NUMERATOR(IZH_A_NUMERATOR),
                .A_DENOMINATOR(IZH_A_DENOMINATOR),
                .B_NUMERATOR(IZH_B_NUMERATOR),
                .B_DENOMINATOR(IZH_B_DENOMINATOR),
                .RESET_VOLTAGE(IZH_RESET_VOLTAGE),
                .RECOVERY_INCREMENT(IZH_RECOVERY_INCREMENT),
                .SPIKE_THRESHOLD(IZH_SPIKE_THRESHOLD)
            ) u_shared_izhikevich (
                .clk(clk),
                .rst(rst),
                .clear(clear),
                .state_load(izh_state == IZH_LOAD),
                .state_voltage(membrane[neuron_index]),
                .state_recovery(recovery_state[neuron_index]),
                .start(izh_state == IZH_START),
                .syn_current(syn_current[neuron_index]),
                .done(engine_done),
                .spike(engine_spike),
                .membrane(engine_voltage),
                .recovery(engine_recovery)
            );
        end else if (NEURON_MODEL == NEURON_MODEL_HODGKIN_HUXLEY) begin : gen_hh_serial
            localparam int unsigned INDEX_WIDTH =
                (NUM_NEURONS <= 1) ? 1 : $clog2(NUM_NEURONS);
            typedef enum logic [2:0] {
                HH_IDLE,
                HH_LOAD,
                HH_START,
                HH_WAIT,
                HH_DONE
            } hh_state_t;

            hh_state_t hh_state, hh_next_state;
            logic [INDEX_WIDTH-1:0] neuron_index;
            logic [19:0] sodium_activation_state [NUM_NEURONS];
            logic [19:0] sodium_inactivation_state [NUM_NEURONS];
            logic [19:0] potassium_activation_state [NUM_NEURONS];
            logic engine_done;
            logic engine_spike;
            logic signed [MEM_WIDTH-1:0] engine_voltage;
            logic [19:0] engine_m;
            logic [19:0] engine_h;
            logic [19:0] engine_n;
            integer reset_index;

            assign done = (hh_state == HH_DONE);

            always_comb begin
                hh_next_state = hh_state;
                case (hh_state)
                    HH_IDLE: if (update_valid) hh_next_state = HH_LOAD;
                    HH_LOAD: hh_next_state = HH_START;
                    HH_START: hh_next_state = HH_WAIT;
                    HH_WAIT: begin
                        if (engine_done) begin
                            if (neuron_index == NUM_NEURONS - 1)
                                hh_next_state = HH_DONE;
                            else
                                hh_next_state = HH_LOAD;
                        end
                    end
                    HH_DONE: hh_next_state = HH_IDLE;
                    default: hh_next_state = HH_IDLE;
                endcase
            end

            always_ff @(posedge clk or posedge rst) begin
                if (rst || clear) begin
                    hh_state <= HH_IDLE;
                    neuron_index <= '0;
                    spike <= '0;
                    for (reset_index = 0; reset_index < NUM_NEURONS;
                            reset_index = reset_index + 1) begin
                        membrane[reset_index] <= 24'sd0;
                        sodium_activation_state[reset_index] <= 20'd13876;
                        sodium_inactivation_state[reset_index] <= 20'd156269;
                        potassium_activation_state[reset_index] <= 20'd83277;
                    end
                end else begin
                    hh_state <= hh_next_state;
                    if (hh_state == HH_IDLE && update_valid) begin
                        neuron_index <= '0;
                        spike <= '0;
                    end
                    if (hh_state == HH_WAIT && engine_done) begin
                        membrane[neuron_index] <= engine_voltage;
                        sodium_activation_state[neuron_index] <= engine_m;
                        sodium_inactivation_state[neuron_index] <= engine_h;
                        potassium_activation_state[neuron_index] <= engine_n;
                        spike[neuron_index] <= engine_spike;
                        if (neuron_index != NUM_NEURONS - 1)
                            neuron_index <= neuron_index + 1'b1;
                    end
                end
            end

            hodgkin_huxley_neuron #(
                .MEM_WIDTH(MEM_WIDTH),
                .CURRENT_WIDTH(CURRENT_WIDTH),
                .LUT_DIR(HH_LUT_DIR)
            ) u_shared_hh (
                .clk(clk),
                .rst(rst),
                .clear(clear),
                .state_load(hh_state == HH_LOAD),
                .state_voltage(membrane[neuron_index]),
                .state_sodium_activation(sodium_activation_state[neuron_index]),
                .state_sodium_inactivation(sodium_inactivation_state[neuron_index]),
                .state_potassium_activation(potassium_activation_state[neuron_index]),
                .start(hh_state == HH_START),
                .syn_current(syn_current[neuron_index]),
                .done(engine_done),
                .spike(engine_spike),
                .membrane(engine_voltage),
                .sodium_activation(engine_m),
                .sodium_inactivation(engine_h),
                .potassium_activation(engine_n)
            );
        end else begin : gen_parallel
            logic [NUM_NEURONS-1:0] neuron_done;
            logic neuron_start;
            logic all_done;
            typedef enum logic [1:0] {IDLE, START, WAIT, DONE} state_t;
            state_t state, next_state;

            assign all_done = &neuron_done;
            assign neuron_start = (state == START);
            assign done = (state == DONE);

            always_ff @(posedge clk or posedge rst) begin
                if (rst || clear)
                    state <= IDLE;
                else
                    state <= next_state;
            end

            always_comb begin
                next_state = state;
                case (state)
                    IDLE: if (update_valid) next_state = START;
                    START: next_state = WAIT;
                    WAIT: if (all_done) next_state = DONE;
                    DONE: next_state = IDLE;
                    default: next_state = IDLE;
                endcase
            end

            for (genvar i = 0; i < NUM_NEURONS; i++) begin : gen_neurons
                neuron_core #(
                    .NEURON_MODEL(NEURON_MODEL),
                    .MEM_WIDTH(MEM_WIDTH),
                    .CURRENT_WIDTH(CURRENT_WIDTH),
                    .THRESHOLD(THRESHOLD),
                    .RESET_POT(RESET_POT),
                    .LEAK_SHIFT(LEAK_SHIFT),
                    .IZH_FRAC_BITS(IZH_FRAC_BITS),
                    .IZH_A_NUMERATOR(IZH_A_NUMERATOR),
                    .IZH_A_DENOMINATOR(IZH_A_DENOMINATOR),
                    .IZH_B_NUMERATOR(IZH_B_NUMERATOR),
                    .IZH_B_DENOMINATOR(IZH_B_DENOMINATOR),
                    .IZH_RESET_VOLTAGE(IZH_RESET_VOLTAGE),
                    .IZH_RECOVERY_INCREMENT(IZH_RECOVERY_INCREMENT),
                    .IZH_SPIKE_THRESHOLD(IZH_SPIKE_THRESHOLD),
                    .HH_LUT_DIR(HH_LUT_DIR)
                ) u_neuron_core (
                    .clk(clk),
                    .rst(rst),
                    .clear(clear),
                    .start(neuron_start),
                    .syn_current(syn_current[i]),
                    .spike(spike[i]),
                    .membrane(membrane[i]),
                    .done(neuron_done[i])
                );
            end
        end
    endgenerate
endmodule
