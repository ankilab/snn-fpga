/*
------------------------------------------------------------------------------
Module: weight_controller
Authors: MD & ChatGPT
Date: 16.07.2026

Description:
Generates BRAM addresses required to fetch all outgoing synaptic weights
associated with a spiking pre-synaptic neuron. Handles BRAM latency and
iteration across post-synaptic neurons.

------------------------------------------------------------------------------
*/

module weight_controller #(
    parameter int unsigned PRE_NEURONS,
    parameter int unsigned POST_NEURONS,
    parameter int unsigned PRE_WIDTH =
        (PRE_NEURONS <= 1) ? 1 : $clog2(PRE_NEURONS),
    parameter int unsigned POST_WIDTH =
        (POST_NEURONS <= 1) ? 1 : $clog2(POST_NEURONS)
) (
    // INPUTS
    input  logic clk,
    input  logic rst,

    input  logic start,

    input  logic [PRE_WIDTH-1:0] pre_idx,

    // OUTPUTS
    output logic weight_valid,

    output logic [PRE_WIDTH-1:0] bram_pre,
    output logic [POST_WIDTH-1:0] bram_post,
    output logic [POST_WIDTH-1:0] weight_post,

    output logic done
);

    localparam logic [POST_WIDTH-1:0] LAST_POST = POST_WIDTH'(POST_NEURONS-1);

    logic active;
    logic [PRE_WIDTH-1:0] pre_reg;
    logic [POST_WIDTH-1:0] post_counter;

    logic weight_valid_d;
    logic last_weight_d;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            active         <= 1'b0;
            pre_reg        <= '0;
            post_counter   <= '0;
            weight_post    <= '0;
            weight_valid_d <= 1'b0;
            last_weight_d  <= 1'b0;
            done           <= 1'b0;
        end
        else begin
            done <= 1'b0;

            if (start) begin
                active         <= 1'b1;
                pre_reg        <= pre_idx;
                post_counter   <= '0;
                weight_valid_d <= 1'b0;
                last_weight_d  <= 1'b0;
            end
            else if (active) begin
                weight_valid_d <= 1'b1;
                last_weight_d  <= (post_counter == LAST_POST);
                weight_post    <= post_counter;

                if (post_counter == LAST_POST)
                    active <= 1'b0;
                else
                    post_counter <= post_counter + 1'b1;
            end
            else begin
                if (last_weight_d)
                    done <= 1'b1;

                weight_valid_d <= 1'b0;
                last_weight_d  <= 1'b0;
            end
        end
    end

    assign bram_pre     = pre_reg;
    assign bram_post    = post_counter;
    assign weight_valid = weight_valid_d;

endmodule
