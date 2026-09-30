/*
------------------------------------------------------------------------------
Module: synapse_bram
Authors: MD & ChatGPT
Date: 16.07.2026

Description:
Implements BRAM-backed storage for synaptic weights. Supports read and
write operations for individual synapses using pre- and post-neuron
indices.

------------------------------------------------------------------------------
*/

module synapse_bram #(
    parameter int unsigned PRE_NEURONS  = 128,
    parameter int unsigned POST_NEURONS = 128,
    parameter int unsigned WEIGHT_WIDTH = 16,
    // Keep each HDL variable below Vivado's one-million-bit elaboration limit.
    parameter int unsigned MAX_BANK_WORDS = 65536,
    parameter int unsigned PRE_WIDTH =
        (PRE_NEURONS <= 1) ? 1 : $clog2(PRE_NEURONS),
    parameter int unsigned POST_WIDTH =
        (POST_NEURONS <= 1) ? 1 : $clog2(POST_NEURONS)
) (
    // INPUTS
    input logic clk,

    input logic write_en,

    input logic [PRE_WIDTH-1:0] pre_idx,
    input logic [POST_WIDTH-1:0] post_idx,

    input logic signed [WEIGHT_WIDTH-1:0] write_weight,

    // OUTPUTS
    output logic signed [WEIGHT_WIDTH-1:0] read_weight
);

    localparam int unsigned TOTAL_WORDS = PRE_NEURONS * POST_NEURONS;
    localparam int unsigned BANK_WORDS =
        (TOTAL_WORDS < MAX_BANK_WORDS) ? TOTAL_WORDS : MAX_BANK_WORDS;
    localparam int unsigned NUM_BANKS =
        (TOTAL_WORDS + BANK_WORDS - 1) / BANK_WORDS;
    localparam int unsigned LINEAR_ADDR_WIDTH =
        (TOTAL_WORDS <= 1) ? 1 : $clog2(TOTAL_WORDS);
    localparam int unsigned BANK_ADDR_WIDTH =
        (BANK_WORDS <= 1) ? 1 : $clog2(BANK_WORDS);
    localparam int unsigned BANK_INDEX_WIDTH =
        (NUM_BANKS <= 1) ? 1 : $clog2(NUM_BANKS);

    logic [LINEAR_ADDR_WIDTH-1:0] linear_addr;
    logic [BANK_INDEX_WIDTH-1:0] bank_select;
    logic [BANK_ADDR_WIDTH-1:0] bank_addr;
    logic [BANK_INDEX_WIDTH-1:0] read_bank_q;
    logic signed [WEIGHT_WIDTH-1:0] bank_read_data [0:NUM_BANKS-1];

    always_comb begin
        linear_addr = LINEAR_ADDR_WIDTH'(
            (LINEAR_ADDR_WIDTH'(pre_idx) * POST_NEURONS) +
            LINEAR_ADDR_WIDTH'(post_idx)
        );
        bank_select = BANK_INDEX_WIDTH'(linear_addr >> BANK_ADDR_WIDTH);
        bank_addr = BANK_ADDR_WIDTH'(linear_addr);
    end

    initial begin
        if (MAX_BANK_WORDS == 0 ||
            (MAX_BANK_WORDS & (MAX_BANK_WORDS - 1)) != 0)
            $error("MAX_BANK_WORDS must be a nonzero power of two");
        if (BANK_WORDS * WEIGHT_WIDTH >= 1000000)
            $error("synapse bank exceeds Vivado's HDL variable size limit");
    end

    always_ff @(posedge clk) begin
        read_bank_q <= bank_select;
    end

    for (genvar bank_index = 0; bank_index < NUM_BANKS; bank_index++) begin : g_bank
        (* ram_style = "block" *)
        logic signed [WEIGHT_WIDTH-1:0] mem [0:BANK_WORDS-1];

        always_ff @(posedge clk) begin
            if (write_en && bank_select == bank_index)
                mem[bank_addr] <= write_weight;
            bank_read_data[bank_index] <= mem[bank_addr];
        end
    end

    assign read_weight = bank_read_data[read_bank_q];

endmodule
