/*
------------------------------------------------------------------------------
Module: synapse_accumulator
Authors: MD & ChatGPT
Date: 16.07.2026

Description:
Accumulates synaptic currents for all post-synaptic neurons by adding the
weights associated with incoming spike events. Produces the I_syn values
used during neuron updates. I_syn[post] += weight

------------------------------------------------------------------------------
*/

module synapse_accumulator #(
    parameter POST_NEURONS,
    parameter WEIGHT_WIDTH,
    parameter CURRENT_WIDTH,
    parameter POST_WIDTH = (POST_NEURONS <= 1) ? 1 : $clog2(POST_NEURONS)
) (
    // INPUTS
    input logic clk,
    input logic rst,

    // Clears currents after a timestep has been applied to the neurons.
    input logic clear,

    input logic weight_valid,

    input logic signed [WEIGHT_WIDTH-1:0] weight,
    input logic [POST_WIDTH-1:0] post_idx,

    // OUTPUTS
    output logic signed [CURRENT_WIDTH-1:0]
        syn_current [POST_NEURONS]
);

    integer i;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < POST_NEURONS; i++)
                syn_current[i] <= '0;
        end
        else begin
            if (clear) begin
                for (i = 0; i < POST_NEURONS; i++)
                    syn_current[i] <= '0;
            end
            else if (weight_valid) begin
                syn_current[post_idx]
                    <= syn_current[post_idx] + CURRENT_WIDTH'(weight);

            end
        end
    end

endmodule
