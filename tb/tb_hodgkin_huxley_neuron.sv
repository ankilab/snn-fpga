`timescale 1ns/1ps

module tb_hodgkin_huxley_neuron;
    logic clk = 0;
    logic rst = 1;
    logic clear = 0;
    logic start = 0;
    logic signed [23:0] syn_current = 0;
    logic done;
    logic spike;
    logic signed [23:0] membrane;

    integer expected_voltage [0:9];
    integer expected_m [0:9];
    integer expected_h [0:9];
    integer expected_n [0:9];
    integer expected_spike [0:9];

    neuron_core #(
        .NEURON_MODEL(2),
        .MEM_WIDTH(24),
        .CURRENT_WIDTH(24)
    ) dut (
        .clk(clk), .rst(rst), .clear(clear), .start(start),
        .syn_current(syn_current), .done(done), .spike(spike),
        .membrane(membrane)
    );

    always #5 clk = ~clk;

    task automatic update(input integer current);
        begin
            @(negedge clk);
            syn_current = current;
            start = 1;
            @(negedge clk);
            start = 0;
            while (!done)
                @(negedge clk);
        end
    endtask

    initial begin
        expected_voltage[0] = -192;   expected_m[0] = 13483;  expected_h[0] = 156519; expected_n[0] = 83084;  expected_spike[0] = 0;
        expected_voltage[1] = -114;   expected_m[1] = 13483;  expected_h[1] = 156719; expected_n[1] = 82984;  expected_spike[1] = 0;
        expected_voltage[2] = 15;     expected_m[2] = 13632;  expected_h[2] = 156886; expected_n[2] = 82906;  expected_spike[2] = 0;
        expected_voltage[3] = 149;    expected_m[3] = 13865;  expected_h[3] = 156786; expected_n[3] = 83004;  expected_spike[3] = 0;
        expected_voltage[4] = 187;    expected_m[4] = 13865;  expected_h[4] = 156693; expected_n[4] = 83004;  expected_spike[4] = 0;
        expected_voltage[5] = 37070;  expected_m[5] = 28265;  expected_h[5] = 151429; expected_n[5] = 86368;  expected_spike[5] = 0;
        expected_voltage[6] = 359109; expected_m[6] = 176490; expected_h[6] = 106499; expected_n[6] = 112977; expected_spike[6] = 1;
        expected_voltage[7] = 284407; expected_m[7] = 259292; expected_h[7] = 39263;  expected_n[7] = 191556; expected_spike[7] = 0;
        expected_voltage[8] = 93323;  expected_m[8] = 200157; expected_h[8] = 19991;  expected_n[8] = 201032; expected_spike[8] = 0;
        expected_voltage[9] = -41282; expected_m[9] = 5463;   expected_h[9] = 37482;  expected_n[9] = 180987; expected_spike[9] = 0;

        repeat (2) @(negedge clk);
        rst = 0;
        @(negedge clk);
        if (membrane !== 0
                || dut.gen_hh.u_hodgkin_huxley_neuron.sodium_activation !== 13876
                || dut.gen_hh.u_hodgkin_huxley_neuron.sodium_inactivation !== 156269
                || dut.gen_hh.u_hodgkin_huxley_neuron.potassium_activation !== 83277)
            $fatal(1, "incorrect HH resting state");

        for (integer timestep = 0; timestep < 10; timestep = timestep + 1) begin
            update(timestep >= 5 ? 10 : 0);
            if ($signed(membrane) !== expected_voltage[timestep])
                $fatal(1, "voltage mismatch at %0d: got %0d expected %0d",
                    timestep, $signed(membrane), expected_voltage[timestep]);
            if (dut.gen_hh.u_hodgkin_huxley_neuron.sodium_activation !== expected_m[timestep])
                $fatal(1, "m mismatch at %0d", timestep);
            if (dut.gen_hh.u_hodgkin_huxley_neuron.sodium_inactivation !== expected_h[timestep])
                $fatal(1, "h mismatch at %0d", timestep);
            if (dut.gen_hh.u_hodgkin_huxley_neuron.potassium_activation !== expected_n[timestep])
                $fatal(1, "n mismatch at %0d", timestep);
            if (spike !== expected_spike[timestep])
                $fatal(1, "spike mismatch at %0d", timestep);
        end

        @(negedge clk);
        clear = 1;
        @(negedge clk);
        clear = 0;
        if (membrane !== 0)
            $fatal(1, "clear did not restore HH voltage");
        $display("PASS: Hodgkin-Huxley RTL matches ten bit-exact Python updates");
        $finish;
    end
endmodule
