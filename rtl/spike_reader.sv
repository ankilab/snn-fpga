/*
------------------------------------------------------------------------------
Module: spike_reader
Authors: MD & ChatGPT
Date: 16.07.2026

Description:
Sequentially reads spike events from spike_memory and presents them to the
event-driven processing pipeline. Maintains the current event pointer and
signals when all input events have been consumed.
------------------------------------------------------------------------------
*/

module spike_reader #(
    parameter int NUM_EVENTS      = 1024,
    parameter int TIME_WIDTH      = 32,
    parameter int NEURON_ID_WIDTH = 8,
    parameter int ADDR_WIDTH = (NUM_EVENTS <= 1) ? 1 : $clog2(NUM_EVENTS),
    parameter int EVENT_COUNT_WIDTH =
        (NUM_EVENTS <= 1) ? 1 : $clog2(NUM_EVENTS + 1)
) (
    // INPUTS
    input logic clk,
    input logic rst,

    input logic next_event,
    input logic restart_stream,
    input logic [EVENT_COUNT_WIDTH-1:0] event_count,

    // Configuration write port for loading the input event stream.
    input logic config_write,
    input logic [ADDR_WIDTH-1:0] config_addr,
    input logic [TIME_WIDTH-1:0] config_timestamp,
    input logic [NEURON_ID_WIDTH-1:0] config_neuron_id,

    // OUTPUTS
    output logic event_valid,

    output logic [TIME_WIDTH-1:0]      timestamp,
    output logic [NEURON_ID_WIDTH-1:0] neuron_id,

    output logic end_of_stream
);

    logic [ADDR_WIDTH-1:0] event_ptr;
    logic event_valid_reg;
    logic stream_done;

    //--------------------------------------------------------------------------
    // Event pointer management
    //--------------------------------------------------------------------------

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            event_ptr           <= '0;
            event_valid_reg     <= 1'b0;
            stream_done         <= 1'b0;
        end
        else if (restart_stream) begin
            event_ptr       <= '0;
            event_valid_reg <= (event_count != '0);
            stream_done     <= (event_count == '0);
        end
        else if (!stream_done) begin
            if (event_count == '0) begin
                event_valid_reg <= 1'b0;
                stream_done <= 1'b1;
            end
            else if (!event_valid_reg) begin
                // First event becomes available after reset.
                event_valid_reg <= 1'b1;
            end
            else if (next_event) begin
                // Controller wants next event.
                if (event_ptr != ADDR_WIDTH'(event_count - 1'b1)) begin
                    // Increase pointer as long as last event was not reached.
                    event_ptr <= event_ptr + 1'b1;
                end
                else begin
                    // Last event has been consumed.
                    event_valid_reg <= 1'b0;
                    stream_done     <= 1'b1;
                end
            end
        end
    end

    //--------------------------------------------------------------------------
    // Status signals
    //--------------------------------------------------------------------------

    assign event_valid   = event_valid_reg;
    assign end_of_stream = 
        event_valid_reg && 
        next_event && 
        (event_ptr == ADDR_WIDTH'(event_count - 1'b1));

    //--------------------------------------------------------------------------
    // Memory instance
    //--------------------------------------------------------------------------

    spike_memory #(
        .NUM_EVENTS(NUM_EVENTS),
        .TIME_WIDTH(TIME_WIDTH),
        .NEURON_ID_WIDTH(NEURON_ID_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) mem_i (
        .clk(clk),

        // Writes disabled during inference.
        .write_en(config_write),
        .write_addr(config_addr),
        .write_time(config_timestamp),
        .write_neuron(config_neuron_id),

        .read_addr(event_ptr),

        .read_time(timestamp),
        .read_neuron(neuron_id)
    );

endmodule
