/*
------------------------------------------------------------------------------
Module: snn_top
Authors: MD & ChatGPT
Date: 31.07.2026

Description:
    Top-level wrapper integrating all SNN subsystems into a complete event-
    driven inference accelerator.

Notes on Version 1:
- single-layer, event-driven inference
- spike_memory is not instantiated here because, on AMD/Xilinx devices,
    you'll likely infer BRAM initialized from a .mem file and hide it 
    inside spike_reader.
- output_decoder currently assumes:
    - output neuron 0 = class 0
    - output neuron 1 = class 1
    - etc.

snn_top
 |
 +-- spike_reader
 |
 +-- spike_router
 |
 +-- snn_controller + time_controller
 |       |
 |       +-- snn_layer[0]  (784 -> 256)
 |       |
 |       +-- spike_event_encoder
 |       |
 |       +-- snn_layer[1]  (256 -> 10)
 |
 +-- output_decoder
 |
 +-- prediction_selector
 |
 +-- output_interface
------------------------------------------------------------------------------
*/

module snn_top #(
    parameter int unsigned NUM_EVENTS      = 1024,
    parameter int unsigned INPUT_NEURONS   = 784,
    parameter int unsigned HIDDEN_NEURONS  = 256,
    parameter int unsigned OUTPUT_NEURONS  = 10,
    parameter int unsigned INPUT_WIDTH =
        (INPUT_NEURONS <= 1) ? 1 : $clog2(INPUT_NEURONS),
    parameter int unsigned HIDDEN_WIDTH =
        (HIDDEN_NEURONS <= 1) ? 1 : $clog2(HIDDEN_NEURONS),
    parameter int unsigned OUTPUT_WIDTH =
        (OUTPUT_NEURONS <= 1) ? 1 : $clog2(OUTPUT_NEURONS),
    parameter int unsigned CONFIG_PRE_WIDTH =
        (INPUT_WIDTH > HIDDEN_WIDTH) ? INPUT_WIDTH : HIDDEN_WIDTH,
    parameter int unsigned CONFIG_POST_WIDTH =
        (HIDDEN_WIDTH > OUTPUT_WIDTH) ? HIDDEN_WIDTH : OUTPUT_WIDTH,

    parameter int unsigned TIME_WIDTH      = 32,
    // IMPORTANT: MEM_WIDTH == CURRENT_WIDTH
    // current design of lif_neurons only works with
    // same with membrane and current registers!
    parameter int unsigned MEM_WIDTH       = 24,
    parameter int unsigned CURRENT_WIDTH   = 24,
    parameter int unsigned WEIGHT_WIDTH    = 8,
    parameter int unsigned COUNT_WIDTH     = 16,
    parameter int unsigned HIDDEN_NEURON_MODEL = 0,
    parameter int unsigned OUTPUT_NEURON_MODEL = 0,
    parameter int unsigned HIDDEN_EXECUTOR_COUNT =
        (HIDDEN_NEURON_MODEL == 0) ? HIDDEN_NEURONS : 1,
    parameter int unsigned OUTPUT_EXECUTOR_COUNT =
        (OUTPUT_NEURON_MODEL == 0) ? OUTPUT_NEURONS : 1,

    // IMPORTANT: THRESHOLD & RESET_POT NEED TO MATCH
    // MEM_WIDTH !
    parameter logic signed [MEM_WIDTH-1:0] THRESHOLD = 100,
    parameter logic signed [MEM_WIDTH-1:0] RESET_POT = 0,
    parameter int unsigned LEAK_SHIFT = 3,
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

    input logic start_inference,
    input logic [((NUM_EVENTS <= 1) ? 1 : $clog2(NUM_EVENTS + 1))-1:0]
        input_event_count,
    input logic [TIME_WIDTH-1:0] timestep_count,

    // Synchronous weight configuration interface.
    input logic weight_write_valid,
    input logic weight_layer,
    input logic [CONFIG_PRE_WIDTH-1:0] weight_pre_idx,
    input logic [CONFIG_POST_WIDTH-1:0] weight_post_idx,
    input logic signed [WEIGHT_WIDTH-1:0] weight_write_data,
    output logic weight_write_ready,

    // Synchronous input-spike configuration interface.
    input logic spike_write_valid,
    input logic [((NUM_EVENTS <= 1) ? 1 : $clog2(NUM_EVENTS))-1:0] spike_write_addr,
    input logic [TIME_WIDTH-1:0] spike_write_timestamp,
    input logic [INPUT_WIDTH-1:0] spike_write_neuron_id,
    output logic spike_write_ready,

    // OUTPUTS
    output logic inference_done,
    output logic tx_valid,
    output logic [OUTPUT_WIDTH-1:0] tx_data,

    // Per-inference activity and phase telemetry. Counters clear on START and
    // remain stable after DONE so software can read an exact execution snapshot.
    output logic [63:0] inference_cycles,
    output logic [31:0] consumed_input_events,
    output logic [31:0] emitted_hidden_spikes,
    output logic [31:0] emitted_output_spikes,
    output logic [63:0] synapse_phase_cycles,
    output logic [63:0] hidden_update_phase_cycles,
    output logic [63:0] hidden_routing_phase_cycles,
    output logic [63:0] output_update_phase_cycles,
    output logic [63:0] control_overhead_phase_cycles
);

    //----------------------------------------------------------------------
    // Local Parameters
    //----------------------------------------------------------------------

    localparam NEURON_ID_WIDTH = INPUT_WIDTH;

    //----------------------------------------------------------------------
    // Interconnect Signals
    //----------------------------------------------------------------------

    // Event Stream
    // Handshake (Controller & Reader)
    logic read_event; // from snn_controller to spike_reader
    logic route_event;
    logic event_valid; // from spike_reader to snn_controller
    // Control
    logic end_of_stream; // last event in memory reached
    // Event Data
    logic [TIME_WIDTH-1:0] timestamp; // for snn_controller
    logic [NEURON_ID_WIDTH-1:0] neuron_id; // for spike_router

    // Layer Control
    logic update_layer0;
    logic update_layer1;
    logic layer0_ready;
    logic layer1_ready;
    logic layer0_config_ready;
    logic layer1_config_ready;
    logic controller_idle;

    assign weight_write_ready = controller_idle &&
        ((weight_layer == 1'b0) ? layer0_config_ready : layer1_config_ready);
    assign spike_write_ready = controller_idle;

    logic layer0_done;
    logic layer1_done;

    // Encoder
    logic start_encoder;
    logic encoder_done;

    logic encoder_valid;

    logic [HIDDEN_NEURONS-1:0] layer0_spikes;

    logic [HIDDEN_WIDTH-1:0] encoder_neuron_id;

    // Router

    logic route_valid;
    logic [NEURON_ID_WIDTH-1:0] pre_idx;

    // Output Decoder

    logic [OUTPUT_NEURONS-1:0] output_spikes;
    logic start_output_decode;
    logic output_done;

    logic [COUNT_WIDTH-1:0]
          class_count [OUTPUT_NEURONS];

    // Prediction Selector
    logic prediction_valid;
    logic [OUTPUT_WIDTH-1:0] prediction;

    logic telemetry_active;
    logic [63:0] inference_cycle_work;
    logic [2:0] telemetry_phase;

    localparam logic [2:0] PHASE_CONTROL_OVERHEAD = 3'd0;
    localparam logic [2:0] PHASE_SYNAPSE = 3'd1;
    localparam logic [2:0] PHASE_HIDDEN_UPDATE = 3'd2;
    localparam logic [2:0] PHASE_HIDDEN_ROUTING = 3'd3;
    localparam logic [2:0] PHASE_OUTPUT_UPDATE = 3'd4;

    function automatic logic [31:0] count_output_spikes(
        input logic [OUTPUT_NEURONS-1:0] spikes
    );
        logic [31:0] count;
        count = '0;
        for (int index = 0; index < OUTPUT_NEURONS; index++)
            count = count + spikes[index];
        return count;
    endfunction

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            telemetry_active <= 1'b0;
            inference_cycle_work <= '0;
            inference_cycles <= '0;
            consumed_input_events <= '0;
            emitted_hidden_spikes <= '0;
            emitted_output_spikes <= '0;
            synapse_phase_cycles <= '0;
            hidden_update_phase_cycles <= '0;
            hidden_routing_phase_cycles <= '0;
            output_update_phase_cycles <= '0;
            control_overhead_phase_cycles <= '0;
        end else if (start_inference) begin
            telemetry_active <= 1'b1;
            inference_cycle_work <= '0;
            inference_cycles <= '0;
            consumed_input_events <= '0;
            emitted_hidden_spikes <= '0;
            emitted_output_spikes <= '0;
            synapse_phase_cycles <= '0;
            hidden_update_phase_cycles <= '0;
            hidden_routing_phase_cycles <= '0;
            output_update_phase_cycles <= '0;
            control_overhead_phase_cycles <= '0;
        end else if (telemetry_active) begin
            if (inference_done) begin
                telemetry_active <= 1'b0;
                inference_cycles <= inference_cycle_work;
            end else begin
                inference_cycle_work <= inference_cycle_work + 1'b1;
                case (telemetry_phase)
                    PHASE_SYNAPSE:
                        synapse_phase_cycles <= synapse_phase_cycles + 1'b1;
                    PHASE_HIDDEN_UPDATE:
                        hidden_update_phase_cycles <=
                            hidden_update_phase_cycles + 1'b1;
                    PHASE_HIDDEN_ROUTING:
                        hidden_routing_phase_cycles <=
                            hidden_routing_phase_cycles + 1'b1;
                    PHASE_OUTPUT_UPDATE:
                        output_update_phase_cycles <=
                            output_update_phase_cycles + 1'b1;
                    PHASE_CONTROL_OVERHEAD:
                        control_overhead_phase_cycles <=
                            control_overhead_phase_cycles + 1'b1;
                    default:
                        control_overhead_phase_cycles <=
                            control_overhead_phase_cycles + 1'b1;
                endcase
            end

            if (read_event)
                consumed_input_events <= consumed_input_events + 1'b1;
            if (encoder_valid && layer1_ready)
                emitted_hidden_spikes <= emitted_hidden_spikes + 1'b1;
            if (start_output_decode)
                emitted_output_spikes <= emitted_output_spikes +
                    count_output_spikes(output_spikes);
        end
    end

    //----------------------------------------------------------------------
    // Input Spikes
    //----------------------------------------------------------------------

    spike_reader #(
        .NUM_EVENTS(NUM_EVENTS),
        .TIME_WIDTH(TIME_WIDTH),
        .NEURON_ID_WIDTH(NEURON_ID_WIDTH)
    ) u_spike_reader (
        // IN
        .clk(clk),
        .rst(rst),
        .next_event(read_event),
        .restart_stream(start_inference),
        .event_count(input_event_count),
        .config_write(spike_write_valid && spike_write_ready),
        .config_addr(spike_write_addr),
        .config_timestamp(spike_write_timestamp),
        .config_neuron_id(spike_write_neuron_id),

        // OUT
        .event_valid(event_valid),
        .timestamp(timestamp),
        .neuron_id(neuron_id), // important for spike_router
        .end_of_stream(end_of_stream)
    );

    //----------------------------------------------------------------------
    // Spike Routing
    //----------------------------------------------------------------------

    spike_router #(
        .NEURON_ID_WIDTH(NEURON_ID_WIDTH)
    ) u_spike_router (
        // IN
        .event_valid(route_event), // snn_controller checks time and commands go
        .neuron_id(neuron_id),

        // OUT
        .route_valid(route_valid),
        .pre_idx(pre_idx)
    );

    //----------------------------------------------------------------------
    // FIRST LAYER : INPUT to HIDDEN
    //----------------------------------------------------------------------

    snn_layer #(
        .PRE_NEURONS(INPUT_NEURONS),
        .POST_NEURONS(HIDDEN_NEURONS),

        .WEIGHT_WIDTH(WEIGHT_WIDTH),

        .MEM_WIDTH(MEM_WIDTH),
        .CURRENT_WIDTH(CURRENT_WIDTH),
        .NEURON_MODEL(HIDDEN_NEURON_MODEL),
        .EXECUTOR_COUNT(HIDDEN_EXECUTOR_COUNT),

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
    layer0
    (
        .clk(clk),
        .rst(rst),
        .inference_clear(start_inference),

        .pre_valid(route_valid),
        .pre_idx(pre_idx),

        .pre_ready(layer0_ready),
        .update_valid(update_layer0),

        .config_write(weight_write_valid && weight_write_ready && !weight_layer),
        .config_pre_idx(weight_pre_idx[INPUT_WIDTH-1:0]),
        .config_post_idx(weight_post_idx[HIDDEN_WIDTH-1:0]),
        .config_weight(weight_write_data),
        .config_ready(layer0_config_ready),

        .post_spikes(layer0_spikes),

        .done(layer0_done)
    );

    //----------------------------------------------------------------------
    // SPIKE ENCODER : BETWEEN FIRST AND SECOND LAYER
    //----------------------------------------------------------------------
    spike_event_encoder #(
        .NUM_NEURONS(HIDDEN_NEURONS)
    )
    u_hidden_encoder
    (
        .clk(clk),
        .rst(rst),

        .spikes(layer0_spikes),

        .start(start_encoder),

        .event_ready(layer1_ready),

        .event_valid(encoder_valid),

        .neuron_id(encoder_neuron_id),

        .done(encoder_done)
    );

    //----------------------------------------------------------------------
    // SECOND LAYER : HIDDEN to OUTPUT
    //----------------------------------------------------------------------
    snn_layer #(

        .PRE_NEURONS(HIDDEN_NEURONS),
        .POST_NEURONS(OUTPUT_NEURONS),

        .WEIGHT_WIDTH(WEIGHT_WIDTH),

        .MEM_WIDTH(MEM_WIDTH),
        .CURRENT_WIDTH(CURRENT_WIDTH),
        .NEURON_MODEL(OUTPUT_NEURON_MODEL),
        .EXECUTOR_COUNT(OUTPUT_EXECUTOR_COUNT),

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
    layer1
    (
        .clk(clk),
        .rst(rst),
        .inference_clear(start_inference),

        .pre_valid(encoder_valid),
        .pre_idx(encoder_neuron_id),

        .pre_ready(layer1_ready),
        .update_valid(update_layer1),

        .config_write(weight_write_valid && weight_write_ready && weight_layer),
        .config_pre_idx(weight_pre_idx[HIDDEN_WIDTH-1:0]),
        .config_post_idx(weight_post_idx[OUTPUT_WIDTH-1:0]),
        .config_weight(weight_write_data),
        .config_ready(layer1_config_ready),

        .post_spikes(output_spikes),

        .done(layer1_done)
    );

    //----------------------------------------------------------------------
    // Output Decoder + Interface
    //----------------------------------------------------------------------

    output_decoder #(
        .NUM_CLASSES(OUTPUT_NEURONS),
        .COUNT_WIDTH(COUNT_WIDTH)
    ) u_output_decoder (
        // IN
        .clk(clk),
        .rst(rst),
        .clear(start_inference),

        // Version 1:
        // Assumes output neuron index == class ID.
        .enable(start_output_decode),
        .spikes(output_spikes),

        // OUT
        .class_count(class_count),

        .done(output_done)
    );

    prediction_selector #(
        .NUM_CLASSES(OUTPUT_NEURONS),
        .COUNT_WIDTH(COUNT_WIDTH),
        .CLASS_WIDTH(OUTPUT_WIDTH)
    ) u_prediction_selector (
        // IN
        // Export the accumulated argmax exactly once, after the final timestep.
        .enable(inference_done),
        .class_count(class_count),

        // OUT
        .prediction_valid(prediction_valid),
        .prediction(prediction)
    );

    output_interface #(
        .CLASS_WIDTH(OUTPUT_WIDTH)
    ) u_output_interface (
        // IN
        .clk(clk),
        .rst(rst),

        .prediction_valid(prediction_valid),
        .prediction(prediction),

        // OUT
        .tx_valid(tx_valid),
        .tx_data(tx_data)
    );

    //----------------------------------------------------------------------
    // Global Control
    //----------------------------------------------------------------------

    snn_controller #(.TIME_WIDTH(TIME_WIDTH))
    u_snn_controller
    (
        // IN
        .clk(clk),
        .rst(rst),
        .start_inference(start_inference),
        // Event Stream
        .event_valid(event_valid),
        .event_timestamp(timestamp),
        .end_of_stream(end_of_stream),
        .timestep_count(timestep_count),
        .input_event_ready(layer0_ready),
        .layer0_done(layer0_done),
        .encoder_done(encoder_done),
        .layer1_ready(layer1_ready),
        .layer1_done(layer1_done),
        .output_done(output_done),

        // OUT
        // Control Signals
        .read_event(read_event), // to spike_reader
        .route_event(route_event), // to spike_router
        .update_layer0(update_layer0),
        .start_encoder(start_encoder),
        .update_layer1(update_layer1),
        .start_output_decode(start_output_decode), // to output_decoder
        .inference_done(inference_done), // final stop signal
        .controller_idle(controller_idle),
        .telemetry_phase(telemetry_phase)
    );

endmodule
