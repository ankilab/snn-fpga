/*
------------------------------------------------------------------------------
Module: spike_memory
Authors: MD & ChatGPT
Date: 16.07.2026

Description:
Stores pre-generated input spike events in FPGA BRAM. Each event consists
of a timestamp and the identifier of the neuron that emitted the spike.
The module supports initialization and sequential read access during
inference.
------------------------------------------------------------------------------
*/

module spike_memory #(
    parameter int NUM_EVENTS      = 1024,
    parameter int TIME_WIDTH      = 32,
    parameter int NEURON_ID_WIDTH = 8,
    parameter int ADDR_WIDTH = (NUM_EVENTS <= 1) ? 1 : $clog2(NUM_EVENTS)
) (
    // INPUTS
    input logic clk,
    input logic write_en,

    input logic [ADDR_WIDTH-1:0] write_addr,

    input logic [TIME_WIDTH-1:0]      write_time,
    input logic [NEURON_ID_WIDTH-1:0] write_neuron,

    // READ ADDRESS
    input logic [ADDR_WIDTH-1:0] read_addr,

    // OUTPUTS
    output logic [TIME_WIDTH-1:0]      read_time,
    output logic [NEURON_ID_WIDTH-1:0] read_neuron
);

    logic [TIME_WIDTH-1:0] time_mem [0:NUM_EVENTS-1];
    logic [NEURON_ID_WIDTH-1:0] neuron_mem [0:NUM_EVENTS-1];

    // Synchronous write
    always_ff @(posedge clk) begin
        if (write_en) begin
            time_mem[write_addr]   <= write_time;
            neuron_mem[write_addr] <= write_neuron;
        end
    end

    // Combinational read
    assign read_time   = time_mem[read_addr];
    assign read_neuron = neuron_mem[read_addr];

endmodule
