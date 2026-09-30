`timescale 1ns/1ps

module tb_snn_layer;
    localparam int PRE_NEURONS = 2;
    localparam int POST_NEURONS = 2;
    localparam int WIDTH = 16;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic inference_clear = 1'b0;
    logic pre_valid = 1'b0;
    logic pre_idx = 1'b0;
    logic pre_ready;
    logic update_valid = 1'b0;
    logic config_write = 1'b0;
    logic config_pre_idx = 1'b0;
    logic config_post_idx = 1'b0;
    logic signed [7:0] config_weight = '0;
    logic config_ready;
    logic [POST_NEURONS-1:0] post_spikes;
    logic done;

    snn_layer #(
        .PRE_NEURONS(PRE_NEURONS),
        .POST_NEURONS(POST_NEURONS),
        .WEIGHT_WIDTH(8),
        .MEM_WIDTH(WIDTH),
        .CURRENT_WIDTH(WIDTH),
        .THRESHOLD(100),
        .LEAK_SHIFT(2)
    ) dut (.*);

    always #5 clk = ~clk;

    task automatic send_event(input logic idx);
        wait (pre_ready);
        @(negedge clk);
        pre_idx = idx;
        pre_valid = 1'b1;
        @(negedge clk);
        pre_valid = 1'b0;
        wait (!pre_ready);
        wait (pre_ready);
    endtask

    task automatic write_weight(input logic pre, input logic post,
                                input logic signed [7:0] value);
        wait (config_ready);
        @(negedge clk);
        config_pre_idx = pre;
        config_post_idx = post;
        config_weight = value;
        config_write = 1'b1;
        @(negedge clk);
        config_write = 1'b0;
    endtask

    initial begin
        // Two input events should sum to currents [10+3, 20+4].
        repeat (2) @(posedge clk);
        rst <= 1'b0;

        write_weight(0, 0, 8'sd10);
        write_weight(0, 1, 8'sd20);
        write_weight(1, 0, 8'sd3);
        write_weight(1, 1, 8'sd4);

        send_event(0);
        send_event(1);

        if (dut.syn_current[0] !== 16'sd13 ||
            dut.syn_current[1] !== 16'sd24)
            $fatal(1, "accumulated currents=[%0d,%0d], expected [13,24]",
                   dut.syn_current[0], dut.syn_current[1]);

        @(negedge clk);
        update_valid = 1'b1;
        @(negedge clk);
        update_valid = 1'b0;
        wait (done);
        #1;

        if (dut.u_neuron_array.membrane[0] !== 16'sd13)
            $fatal(1, "neuron 0 membrane=%0d, expected 13", dut.u_neuron_array.membrane[0]);
        if (dut.u_neuron_array.membrane[1] !== 16'sd24)
            $fatal(1, "neuron 1 membrane=%0d, expected 24", dut.u_neuron_array.membrane[1]);
        @(posedge clk);
        #1;
        if (dut.syn_current[0] !== 0 || dut.syn_current[1] !== 0)
            $fatal(1, "currents were not cleared after update");

        $display("RESULT: PASS");
        $finish;
    end

    initial begin
        #5000;
        $fatal(1, "timeout");
    end
endmodule
