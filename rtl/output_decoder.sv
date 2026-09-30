/*
------------------------------------------------------------------------------
Module: output_decoder
Authors: MD & ChatGPT
Date: 16.07.2026

Description:
Converts output neuron spike events into class-specific spike counts.
Maintains one counter per output class for the duration of inference.

Version 1 assumptions:
- Output neuron index == class index.
- Only the first NUM_CLASSES neurons are interpreted as output neurons.
- Decoding is performed whenever enable is asserted.
------------------------------------------------------------------------------
*/

module output_decoder #(
    parameter NUM_CLASSES = 10,
    parameter COUNT_WIDTH = 16
) (
    // INPUTS
    input logic clk,
    input logic rst,
    input logic clear,

    input logic enable,

    // Output spikes from neuron_array.
    input logic [NUM_CLASSES-1:0] spikes,

    // OUTPUTS
    output logic [COUNT_WIDTH-1:0] class_count [NUM_CLASSES],

    // One-cycle completion pulse.
    output logic done
);

    integer i;

    always_ff @(posedge clk or posedge rst) begin
        if (rst || clear) begin
            for (i = 0; i < NUM_CLASSES; i++) begin
                class_count[i] <= '0;
            end

            done <= 1'b0;
        end
        else begin

            // Default: no completion pulse.
            done <= 1'b0;

            if (enable) begin

                // Count spikes for each output class.
                for (i = 0; i < NUM_CLASSES; i++) begin
                    if (spikes[i]) begin
                        class_count[i] <= class_count[i] + 1'b1;
                    end
                end

                // Single-cycle handshake back to the controller.
                done <= 1'b1;
            end
        end
    end

endmodule
