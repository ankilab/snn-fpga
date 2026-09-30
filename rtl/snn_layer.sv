/*
------------------------------------------------------------------------------
Module: snn_layer
Authors: MD & ChatGPT
Date: 17.07.2026

Description:
Generic event-driven fully-connected SNN layer.

Implements:
- Converts incoming spike events into accumulated currents.
- Updates a selectable population of neurons
- Produces a spike vector for the next layer

pre spike event
    to
instance: weight_controller
    to
instance: synapse_bram
    to
instance: synapse_accumulator
    to
instance: neuron_array
    to
post spikes


Example configurations: (MNIST LIF SNN)
    784 inputs -> 256 hidden
    256 hidden -> 10 output

------------------------------------------------------------------------------
*/


module snn_layer #(

    parameter int unsigned PRE_NEURONS  = 784,
    parameter int unsigned POST_NEURONS = 256,
    parameter int unsigned PRE_WIDTH =
        (PRE_NEURONS <= 1) ? 1 : $clog2(PRE_NEURONS),
    parameter int unsigned POST_WIDTH =
        (POST_NEURONS <= 1) ? 1 : $clog2(POST_NEURONS),

    parameter int unsigned WEIGHT_WIDTH = 8,

    parameter int unsigned MEM_WIDTH     = 24,
    parameter int unsigned CURRENT_WIDTH = 24,
    parameter int unsigned NEURON_MODEL   = 0,
    parameter int unsigned EXECUTOR_COUNT =
        (NEURON_MODEL == 0) ? POST_NEURONS : 1,

    parameter logic signed [MEM_WIDTH-1:0] THRESHOLD = 100,
    parameter logic signed [MEM_WIDTH-1:0] RESET_POT = 0,
    parameter LEAK_SHIFT = 3,
    parameter int unsigned IZH_FRAC_BITS = 8,
    parameter int unsigned IZH_A_NUMERATOR = 1,
    parameter int unsigned IZH_A_DENOMINATOR = 50,
    parameter int unsigned IZH_B_NUMERATOR = 1,
    parameter int unsigned IZH_B_DENOMINATOR = 5,
    parameter integer IZH_RESET_VOLTAGE = -65,
    parameter integer IZH_RECOVERY_INCREMENT = 8,
    parameter integer IZH_SPIKE_THRESHOLD = 30,
    parameter HH_LUT_DIR = "data/hh_luts/"

)(  
    // INPUTS
    input logic clk,
    input logic rst,
    input logic inference_clear,

    // Incoming spike event. An event is accepted on pre_valid && pre_ready.
    input logic pre_valid,
    input logic [PRE_WIDTH-1:0] pre_idx,

    output logic pre_ready,

    // Applies all currents accumulated in the current timestep.
    input logic update_valid,

    // Configuration write port. Use only while the layer is idle.
    input logic config_write,
    input logic [PRE_WIDTH-1:0] config_pre_idx,
    input logic [POST_WIDTH-1:0] config_post_idx,
    input logic signed [WEIGHT_WIDTH-1:0] config_weight,
    output logic config_ready,

    // OUTPUTS
    // Output spikes
    output logic [POST_NEURONS-1:0] post_spikes,
    // Layer completion pulse
    output logic done
);
    //----------------------------------------------------------------------
    // Internal signals
    //----------------------------------------------------------------------

    //----------------------------------------------------------------------
    // Weight controller signals
    //----------------------------------------------------------------------

    logic weight_valid;

    logic [PRE_WIDTH-1:0]
        bram_pre;

    logic [POST_WIDTH-1:0]
        bram_post;

    logic [POST_WIDTH-1:0]
        weight_post;

    logic weight_fetch_done;

    //----------------------------------------------------------------------
    // BRAM signals
    //----------------------------------------------------------------------

    logic signed [WEIGHT_WIDTH-1:0]
        read_weight;

    logic [PRE_WIDTH-1:0] memory_pre;
    logic [POST_WIDTH-1:0] memory_post;

    assign memory_pre = config_write ? config_pre_idx : bram_pre;
    assign memory_post = config_write ? config_post_idx : bram_post;

    //----------------------------------------------------------------------
    // Accumulator signals
    //----------------------------------------------------------------------

    logic signed [CURRENT_WIDTH-1:0]
        syn_current [POST_NEURONS];

    //----------------------------------------------------------------------
    // Neuron signals
    //----------------------------------------------------------------------

    logic start_neuron_update;
    logic neuron_update_done;
    logic [POST_NEURONS-1:0] neuron_spikes;

    // Preserve the one-cycle neuron spike pulse for the encoder/decoder.
    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            post_spikes <= '0;
        else if (inference_clear)
            post_spikes <= '0;
        else if (start_neuron_update)
            post_spikes <= '0;
        else if (|neuron_spikes)
            post_spikes <= neuron_spikes;
    end

    //----------------------------------------------------------------------
    // Layer FSM
    //----------------------------------------------------------------------

    typedef enum logic [2:0] {

        IDLE,
        WAIT_ACCUM,
        WAIT_UPDATE,
        CLEAR_CURRENTS

    } state_t;

    state_t state, next_state;

    assign config_ready = (state == IDLE);


    always_ff @(posedge clk or posedge rst) begin
        if(rst)
            state <= IDLE;
        else if(inference_clear)
            state <= IDLE;
        else
            state <= next_state;
    end

    always_comb begin

        next_state = state;

        case(state)
            IDLE:
            begin
                if(update_valid)
                    next_state = WAIT_UPDATE;
                else if(pre_valid)
                    next_state = WAIT_ACCUM;
            end

            WAIT_ACCUM:
            begin
                if(weight_fetch_done)
                    next_state = IDLE;
            end

            WAIT_UPDATE:
            begin
                if(neuron_update_done)
                    next_state = CLEAR_CURRENTS;
            end

            CLEAR_CURRENTS:
            begin
                next_state = IDLE;
            end

            default:
                next_state = IDLE;
        endcase
    end

    //----------------------------------------------------------------------
    // Control pulse generation
    //----------------------------------------------------------------------

    always_comb begin
        start_neuron_update = 1'b0;

        pre_ready = 1'b0;

        done = 1'b0;

        case(state)

            IDLE:
            begin
                pre_ready = !update_valid;
                start_neuron_update = update_valid;
            end

            CLEAR_CURRENTS:
                done = 1'b1;

            default:
            begin
            end
        endcase
    end

    //----------------------------------------------------------------------
    // Weight controller
    //----------------------------------------------------------------------

    weight_controller #(
        .PRE_NEURONS(PRE_NEURONS),
        .POST_NEURONS(POST_NEURONS)
    )
    u_weight_controller
    (   
        // IN
        .clk(clk),
        .rst(rst),

        .start(pre_valid && pre_ready),
        .pre_idx(pre_idx),

        // OUT
        .weight_valid(weight_valid),

        .bram_pre(bram_pre),
        .bram_post(bram_post),
        .weight_post(weight_post),

        .done(weight_fetch_done)
    );

    //----------------------------------------------------------------------
    // Synaptic weight memory
    //----------------------------------------------------------------------

    synapse_bram #(
        .PRE_NEURONS(PRE_NEURONS),
        .POST_NEURONS(POST_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH)
    )
    u_synapse_bram
    (
        // IN
        .clk(clk),
        .write_en(config_write && config_ready),

        .pre_idx(memory_pre),
        .post_idx(memory_post),

        .write_weight(config_weight),

        // OUT
        .read_weight(read_weight)
    );

    //----------------------------------------------------------------------
    // Current accumulator
    //----------------------------------------------------------------------

    synapse_accumulator #(
        .POST_NEURONS(POST_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .CURRENT_WIDTH(CURRENT_WIDTH)
    )
    u_synapse_accumulator
    (
        // IN
        .clk(clk),
        .rst(rst),

        .clear(inference_clear || (state == CLEAR_CURRENTS)),

        .weight_valid(weight_valid),

        .weight(read_weight),
        .post_idx(weight_post),

        // OUT
        .syn_current(syn_current)
    );

    //----------------------------------------------------------------------
    // Selectable neuron population
    //----------------------------------------------------------------------

    neuron_array #(
        .NUM_NEURONS(POST_NEURONS),
        .MEM_WIDTH(MEM_WIDTH),
        .CURRENT_WIDTH(CURRENT_WIDTH),
        .NEURON_MODEL(NEURON_MODEL),
        .EXECUTOR_COUNT(EXECUTOR_COUNT),
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
    )
    u_neuron_array
    (
        // IN
        .clk(clk),
        .rst(rst),
        .clear(inference_clear),

        .update_valid(start_neuron_update),

        .syn_current(syn_current),

        // OUT
        .spike(neuron_spikes),
        .membrane(),

        .done(neuron_update_done)
    );

endmodule
