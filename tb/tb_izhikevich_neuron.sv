`timescale 1ns/1ps

module tb_izhikevich_neuron;
    localparam int MEM_WIDTH = 24;
    localparam int CURRENT_WIDTH = 24;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic clear = 1'b0;
    logic start = 1'b0;
    logic signed [CURRENT_WIDTH-1:0] syn_current = '0;
    logic done;
    logic spike;
    logic signed [MEM_WIDTH-1:0] membrane;
    wire signed [MEM_WIDTH-1:0] recovery =
        dut.gen_izhikevich.u_izhikevich_neuron.recovery;
    logic spike_seen = 1'b0;

    always #5 clk = ~clk;

    always_ff @(posedge clk) begin
        if (rst || clear)
            spike_seen <= 1'b0;
        else if (spike)
            spike_seen <= 1'b1;
    end

    neuron_core #(
        .NEURON_MODEL(1),
        .MEM_WIDTH(MEM_WIDTH),
        .CURRENT_WIDTH(CURRENT_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .clear(clear),
        .start(start),
        .syn_current(syn_current),
        .done(done),
        .spike(spike),
        .membrane(membrane)
    );

    task automatic run_step(
        input integer current,
        input integer expected_voltage,
        input integer expected_recovery,
        input logic expected_spike
    );
        begin
            @(negedge clk);
            spike_seen = 1'b0;
            syn_current = current;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            wait (done);
            #1;
            if ($signed(membrane) !== expected_voltage)
                $fatal(1, "membrane=%0d expected=%0d", membrane, expected_voltage);
            if ($signed(recovery) !== expected_recovery)
                $fatal(1, "recovery=%0d expected=%0d", recovery, expected_recovery);
            if (spike_seen !== expected_spike)
                $fatal(1, "spike_seen=%0b expected=%0b", spike_seen, expected_spike);
            @(negedge clk);
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        rst = 1'b0;
        #1;
        if ($signed(membrane) !== -16640 || $signed(recovery) !== -3328)
            $fatal(1, "initial state v=%0d u=%0d", membrane, recovery);

        run_step(10, -14875, -3321, 1'b0);
        run_step(10, -12717, -3306, 1'b0);
        run_step(10, -8234, -3273, 1'b0);
        run_step(10, -16640, -1112, 1'b1);

        $display("RESULT: PASS");
        $finish;
    end
endmodule
