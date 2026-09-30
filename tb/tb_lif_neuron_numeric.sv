`timescale 1ns/1ps

module tb_lif_neuron_numeric;
    logic clk = 1'b0;
    logic rst = 1'b1;

    logic overflow_clear = 1'b0;
    logic overflow_start = 1'b0;
    logic signed [7:0] overflow_current = '0;
    logic overflow_done;
    logic overflow_spike;
    logic signed [7:0] overflow_membrane;

    logic negative_clear = 1'b0;
    logic negative_start = 1'b0;
    logic signed [7:0] negative_current = '0;
    logic negative_done;
    logic negative_spike;
    logic signed [7:0] negative_membrane;

    lif_neuron #(
        .MEM_WIDTH(8),
        .CURRENT_WIDTH(8),
        .THRESHOLD(127),
        .RESET_POT(0),
        .LEAK_SHIFT(7)
    ) overflow_dut (
        .clk(clk),
        .rst(rst),
        .clear(overflow_clear),
        .state_load(1'b0),
        .state_membrane('0),
        .start(overflow_start),
        .syn_current(overflow_current),
        .done(overflow_done),
        .spike(overflow_spike),
        .membrane(overflow_membrane)
    );

    lif_neuron #(
        .MEM_WIDTH(8),
        .CURRENT_WIDTH(8),
        .THRESHOLD(127),
        .RESET_POT(0),
        .LEAK_SHIFT(1)
    ) negative_dut (
        .clk(clk),
        .rst(rst),
        .clear(negative_clear),
        .state_load(1'b0),
        .state_membrane('0),
        .start(negative_start),
        .syn_current(negative_current),
        .done(negative_done),
        .spike(negative_spike),
        .membrane(negative_membrane)
    );

    always #5 clk = ~clk;

    task automatic update_overflow(input logic signed [7:0] current);
        @(negedge clk);
        overflow_current = current;
        overflow_start = 1'b1;
        @(negedge clk);
        overflow_start = 1'b0;
        wait (overflow_done);
        #1;
    endtask

    task automatic update_negative(input logic signed [7:0] current);
        @(negedge clk);
        negative_current = current;
        negative_start = 1'b1;
        @(negedge clk);
        negative_start = 1'b0;
        wait (negative_done);
        #1;
    endtask

    initial begin
        repeat (2) @(posedge clk);
        rst = 1'b0;

        update_overflow(8'sd120);
        if (overflow_membrane !== 8'sd120 || overflow_spike)
            $fatal(1, "first update: membrane=%0d spike=%0b, expected 120/0",
                   overflow_membrane, overflow_spike);

        // 120 - (120 >>> 7) + 20 = 140, which wraps to signed -116.
        update_overflow(8'sd20);
        if (overflow_membrane !== -8'sd116 || overflow_spike)
            $fatal(1, "overflow update: membrane=%0d spike=%0b, expected -116/0",
                   overflow_membrane, overflow_spike);

        update_negative(-8'sd3);
        if (negative_membrane !== -8'sd3)
            $fatal(1, "negative first update=%0d, expected -3", negative_membrane);

        // Arithmetic -3 >>> 1 is -2, hence -3 - (-2) = -1.
        update_negative(8'sd0);
        if (negative_membrane !== -8'sd1)
            $fatal(1, "negative leak update=%0d, expected -1", negative_membrane);

        $display("RESULT: PASS");
        $finish;
    end

    initial begin
        #2000;
        $fatal(1, "timeout");
    end
endmodule
