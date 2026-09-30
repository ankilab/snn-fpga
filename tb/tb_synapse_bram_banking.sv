`timescale 1ns/1ps

module tb_synapse_bram_banking;
    localparam int PRE_NEURONS = 8;
    localparam int POST_NEURONS = 8;
    localparam int WEIGHT_WIDTH = 8;
    localparam int PRE_WIDTH = $clog2(PRE_NEURONS);
    localparam int POST_WIDTH = $clog2(POST_NEURONS);

    logic clk = 1'b0;
    logic write_en = 1'b0;
    logic [PRE_WIDTH-1:0] pre_idx = '0;
    logic [POST_WIDTH-1:0] post_idx = '0;
    logic signed [WEIGHT_WIDTH-1:0] write_weight = '0;
    logic signed [WEIGHT_WIDTH-1:0] read_weight;

    always #5 clk = ~clk;

    synapse_bram #(
        .PRE_NEURONS(PRE_NEURONS),
        .POST_NEURONS(POST_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .MAX_BANK_WORDS(16)
    ) dut (
        .clk,
        .write_en,
        .pre_idx,
        .post_idx,
        .write_weight,
        .read_weight
    );

    task automatic write_entry(
        input int unsigned pre,
        input int unsigned post,
        input logic signed [WEIGHT_WIDTH-1:0] value
    );
        pre_idx = PRE_WIDTH'(pre);
        post_idx = POST_WIDTH'(post);
        write_weight = value;
        write_en = 1'b1;
        @(posedge clk);
        #1;
        write_en = 1'b0;
    endtask

    task automatic check_entry(
        input int unsigned pre,
        input int unsigned post,
        input logic signed [WEIGHT_WIDTH-1:0] expected
    );
        pre_idx = PRE_WIDTH'(pre);
        post_idx = POST_WIDTH'(post);
        @(posedge clk);
        #1;
        if (read_weight !== expected) begin
            $error("read (%0d,%0d): expected %0d, got %0d",
                   pre, post, expected, read_weight);
            $fatal(1);
        end
    endtask

    initial begin
        // Linear addresses 15/16 straddle the first bank boundary; 63 is last.
        write_entry(1, 7, -8'sd12);
        write_entry(2, 0, 8'sd37);
        write_entry(7, 7, -8'sd99);

        check_entry(1, 7, -8'sd12);
        check_entry(2, 0, 8'sd37);
        check_entry(7, 7, -8'sd99);

        $display("RESULT: PASS");
        $finish;
    end
endmodule
