/* Two-layer event-driven SNN inference controller. */
module snn_controller #(
    parameter int TIME_WIDTH = 32
)(
    input  logic clk,
    input  logic rst,
    input  logic start_inference,

    input  logic event_valid,
    input  logic [TIME_WIDTH-1:0] event_timestamp,
    input  logic end_of_stream,
    input  logic [TIME_WIDTH-1:0] timestep_count,
    input  logic input_event_ready,

    input  logic layer0_done,
    input  logic encoder_done,
    input  logic layer1_ready,
    input  logic layer1_done,
    input  logic output_done,

    output logic read_event,
    output logic route_event,
    output logic update_layer0,
    output logic start_encoder,
    output logic update_layer1,
    output logic start_output_decode,
    output logic inference_done,
    output logic controller_idle,
    output logic [2:0] telemetry_phase
);
    localparam logic [2:0] PHASE_CONTROL_OVERHEAD = 3'd0;
    localparam logic [2:0] PHASE_SYNAPSE = 3'd1;
    localparam logic [2:0] PHASE_HIDDEN_UPDATE = 3'd2;
    localparam logic [2:0] PHASE_HIDDEN_ROUTING = 3'd3;
    localparam logic [2:0] PHASE_OUTPUT_UPDATE = 3'd4;

    typedef enum logic [3:0] {
        IDLE, READ_EVENTS, DRAIN_INPUT, UPDATE_L0, WAIT_L0, START_ENC, WAIT_ENC,
        DRAIN_L1, UPDATE_L1, WAIT_L1, START_OUTPUT, WAIT_OUTPUT, ADVANCE_TIME, DONE
    } state_t;

    state_t state, next_state;
    logic [TIME_WIDTH-1:0] current_time;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) state <= IDLE;
        else     state <= next_state;
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            current_time <= '0;
        end else begin
            if (state == IDLE && start_inference) begin
                current_time <= '0;
            end
            if (state == ADVANCE_TIME)
                current_time <= current_time + 1'b1;
        end
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:         if (start_inference) next_state = READ_EVENTS;
            READ_EVENTS:  if (!event_valid ||
                              (event_timestamp > current_time) ||
                              (event_valid && input_event_ready &&
                               end_of_stream))
                              next_state = DRAIN_INPUT;
            DRAIN_INPUT:  if (input_event_ready) next_state = UPDATE_L0;
            UPDATE_L0:    next_state = WAIT_L0;
            WAIT_L0:      if (layer0_done) next_state = START_ENC;
            START_ENC:    next_state = WAIT_ENC;
            WAIT_ENC:     if (encoder_done) next_state = DRAIN_L1;
            DRAIN_L1:     if (layer1_ready) next_state = UPDATE_L1;
            UPDATE_L1:    next_state = WAIT_L1;
            WAIT_L1:      if (layer1_done) next_state = START_OUTPUT;
            START_OUTPUT: next_state = WAIT_OUTPUT;
            WAIT_OUTPUT:  if (output_done) next_state = ADVANCE_TIME;
            ADVANCE_TIME: begin
                if (current_time + 1'b1 >= timestep_count) next_state = DONE;
                else                                       next_state = READ_EVENTS;
            end
            DONE:         if (!start_inference) next_state = IDLE;
            default:      next_state = IDLE;
        endcase
    end

    always_comb begin
        read_event = 1'b0;
        route_event = 1'b0;
        update_layer0 = 1'b0;
        start_encoder = 1'b0;
        update_layer1 = 1'b0;
        start_output_decode = 1'b0;
        inference_done = 1'b0;
        controller_idle = 1'b0;
        telemetry_phase = PHASE_CONTROL_OVERHEAD;

        if (state == READ_EVENTS && event_valid && input_event_ready &&
            event_timestamp == current_time) begin
            read_event = 1'b1;
            route_event = 1'b1;
        end
        update_layer0       = (state == UPDATE_L0);
        start_encoder       = (state == START_ENC);
        update_layer1       = (state == UPDATE_L1);
        start_output_decode = (state == START_OUTPUT);
        inference_done      = (state == DONE);
        controller_idle     = (state == IDLE);

        case (state)
            READ_EVENTS,
            DRAIN_INPUT: telemetry_phase = PHASE_SYNAPSE;
            UPDATE_L0,
            WAIT_L0: telemetry_phase = PHASE_HIDDEN_UPDATE;
            START_ENC,
            WAIT_ENC,
            DRAIN_L1: telemetry_phase = PHASE_HIDDEN_ROUTING;
            UPDATE_L1,
            WAIT_L1: telemetry_phase = PHASE_OUTPUT_UPDATE;
            default: telemetry_phase = PHASE_CONTROL_OVERHEAD;
        endcase
    end
endmodule
