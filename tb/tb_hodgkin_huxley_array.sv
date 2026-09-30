`timescale 1ns/1ps

module tb_hodgkin_huxley_array;
    logic clk = 0;
    logic rst = 1;
    logic clear = 0;
    logic update_valid = 0;
    logic signed [23:0] syn_current [2];
    logic [1:0] spike;
    logic signed [23:0] membrane [2];
    logic done;

    neuron_array #(
        .NUM_NEURONS(2),
        .MEM_WIDTH(24),
        .CURRENT_WIDTH(24),
        .NEURON_MODEL(2)
    ) dut (.*);

    always #5 clk = ~clk;

    task automatic update;
        begin
            @(negedge clk);
            update_valid = 1;
            @(negedge clk);
            update_valid = 0;
            wait (done);
            @(negedge clk);
        end
    endtask

    initial begin
        syn_current[0] = 10;
        syn_current[1] = 0;
        repeat (3) @(negedge clk);
        rst = 0;
        update();
        if ($signed(membrane[0]) !== 36511 || $signed(membrane[1]) !== -192)
            $fatal(1, "first serialized update mismatch: V=[%0d,%0d]",
                $signed(membrane[0]), $signed(membrane[1]));
        if (spike !== 0)
            $fatal(1, "unexpected spike in first serialized update");

        syn_current[0] = 0;
        update();
        if ($signed(membrane[0]) !== 83270 || $signed(membrane[1]) !== -114)
            $fatal(1, "state isolation mismatch: V=[%0d,%0d]",
                $signed(membrane[0]), $signed(membrane[1]));

        clear = 1;
        @(negedge clk);
        clear = 0;
        if (membrane[0] !== 0 || membrane[1] !== 0)
            $fatal(1, "clear did not reset serialized HH state");
        $display("PASS: shared HH executor preserves independent neuron states");
        $finish;
    end

    initial begin
        #500000;
        $fatal(1, "timeout");
    end
endmodule
