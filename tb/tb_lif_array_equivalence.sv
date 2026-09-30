`timescale 1ns/1ps

module tb_lif_array_equivalence;
    localparam int NUM_NEURONS = 7;
    localparam int MEM_WIDTH = 16;
    localparam int CURRENT_WIDTH = 16;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic clear = 1'b0;
    logic update_valid = 1'b0;
    logic signed [CURRENT_WIDTH-1:0] syn_current [NUM_NEURONS];
    logic [NUM_NEURONS-1:0] parallel_spike;
    logic [NUM_NEURONS-1:0] serial_spike;
    logic signed [MEM_WIDTH-1:0] parallel_membrane [NUM_NEURONS];
    logic signed [MEM_WIDTH-1:0] serial_membrane [NUM_NEURONS];
    logic parallel_done;
    logic serial_done;
    logic [NUM_NEURONS-1:0] parallel_spike_seen = '0;
    logic [NUM_NEURONS-1:0] serial_spike_seen = '0;
    integer parallel_cycles;
    integer serial_cycles;
    logic measuring = 1'b0;
    logic parallel_completed = 1'b0;
    logic serial_completed = 1'b0;

    neuron_array #(
        .NUM_NEURONS(NUM_NEURONS),
        .MEM_WIDTH(MEM_WIDTH),
        .CURRENT_WIDTH(CURRENT_WIDTH),
        .NEURON_MODEL(0),
        .EXECUTOR_COUNT(NUM_NEURONS),
        .THRESHOLD(100),
        .RESET_POT(0),
        .LEAK_SHIFT(2)
    ) parallel_dut (
        .clk(clk), .rst(rst), .clear(clear),
        .update_valid(update_valid), .syn_current(syn_current),
        .spike(parallel_spike), .membrane(parallel_membrane),
        .done(parallel_done)
    );

    neuron_array #(
        .NUM_NEURONS(NUM_NEURONS),
        .MEM_WIDTH(MEM_WIDTH),
        .CURRENT_WIDTH(CURRENT_WIDTH),
        .NEURON_MODEL(0),
        .EXECUTOR_COUNT(1),
        .THRESHOLD(100),
        .RESET_POT(0),
        .LEAK_SHIFT(2)
    ) serial_dut (
        .clk(clk), .rst(rst), .clear(clear),
        .update_valid(update_valid), .syn_current(syn_current),
        .spike(serial_spike), .membrane(serial_membrane),
        .done(serial_done)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (rst || clear || update_valid) begin
            parallel_spike_seen <= '0;
            serial_spike_seen <= '0;
            parallel_cycles <= 0;
            serial_cycles <= 0;
            parallel_completed <= 1'b0;
            serial_completed <= 1'b0;
        end else begin
            parallel_spike_seen <= parallel_spike_seen | parallel_spike;
            serial_spike_seen <= serial_spike_seen | serial_spike;
            if (measuring && !parallel_completed)
                parallel_cycles <= parallel_cycles + 1;
            if (measuring && !serial_completed)
                serial_cycles <= serial_cycles + 1;
            if (parallel_done)
                parallel_completed <= 1'b1;
            if (serial_done)
                serial_completed <= 1'b1;
        end
    end

    task automatic run_update(input int step);
        for (int index = 0; index < NUM_NEURONS; index++)
            syn_current[index] = ((step * 37 + index * 29) % 221) - 80;

        @(negedge clk);
        measuring = 1'b1;
        update_valid = 1'b1;
        @(negedge clk);
        update_valid = 1'b0;

        wait (parallel_done);
        wait (serial_done);
        @(posedge clk);
        #1;
        measuring = 1'b0;

        for (int index = 0; index < NUM_NEURONS; index++)
            if (serial_membrane[index] !== parallel_membrane[index])
                $fatal(1, "step %0d neuron %0d: serial membrane %0d, parallel %0d",
                       step, index, serial_membrane[index], parallel_membrane[index]);
        if (serial_spike_seen !== parallel_spike_seen)
            $fatal(1, "step %0d: serial spikes %b, parallel %b",
                   step, serial_spike_seen, parallel_spike_seen);
        if (serial_cycles <= parallel_cycles)
            $fatal(1, "serial array must take longer: serial=%0d parallel=%0d",
                   serial_cycles, parallel_cycles);
    endtask

    initial begin
        for (int index = 0; index < NUM_NEURONS; index++)
            syn_current[index] = '0;
        repeat (3) @(posedge clk);
        rst = 1'b0;

        for (int step = 0; step < 16; step++)
            run_update(step);

        @(negedge clk);
        clear = 1'b1;
        @(negedge clk);
        clear = 1'b0;
        @(posedge clk);
        #1;
        for (int index = 0; index < NUM_NEURONS; index++) begin
            if (serial_membrane[index] !== 0 || parallel_membrane[index] !== 0)
                $fatal(1, "clear did not reset neuron %0d", index);
        end

        $display("RESULT: PASS (serial/parallel LIF array equivalence)");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout");
    end
endmodule
