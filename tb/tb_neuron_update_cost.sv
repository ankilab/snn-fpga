`timescale 1ns/1ps

module tb_neuron_update_cost;
    localparam int NUM_NEURONS = 3;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic clear = 1'b0;
    longint unsigned cycle_counter = 0;

    logic lif_start = 0;
    logic izh_start = 0;
    logic hh_start = 0;
    logic signed [23:0] lif_current = 0;
    logic signed [23:0] izh_current = 0;
    logic signed [23:0] hh_current = 0;
    logic lif_done, lif_spike;
    logic izh_done, izh_spike;
    logic hh_done, hh_spike;
    logic signed [23:0] lif_membrane;
    logic signed [23:0] izh_membrane;
    logic signed [23:0] hh_membrane;

    logic lif_array_valid = 0;
    logic izh_array_valid = 0;
    logic hh_array_valid = 0;
    logic signed [23:0] lif_array_current [NUM_NEURONS];
    logic signed [23:0] izh_array_current [NUM_NEURONS];
    logic signed [23:0] hh_array_current [NUM_NEURONS];
    logic [NUM_NEURONS-1:0] lif_array_spike;
    logic [NUM_NEURONS-1:0] izh_array_spike;
    logic [NUM_NEURONS-1:0] hh_array_spike;
    logic signed [23:0] lif_array_membrane [NUM_NEURONS];
    logic signed [23:0] izh_array_membrane [NUM_NEURONS];
    logic signed [23:0] hh_array_membrane [NUM_NEURONS];
    logic lif_array_done, izh_array_done, hh_array_done;

    always #5 clk = ~clk;
    always @(posedge clk)
        cycle_counter <= cycle_counter + 1;

    neuron_core #(.NEURON_MODEL(0)) u_lif (
        .clk(clk), .rst(rst), .clear(clear), .start(lif_start),
        .syn_current(lif_current), .done(lif_done), .spike(lif_spike),
        .membrane(lif_membrane)
    );

    neuron_core #(.NEURON_MODEL(1)) u_izh (
        .clk(clk), .rst(rst), .clear(clear), .start(izh_start),
        .syn_current(izh_current), .done(izh_done), .spike(izh_spike),
        .membrane(izh_membrane)
    );

    neuron_core #(.NEURON_MODEL(2)) u_hh (
        .clk(clk), .rst(rst), .clear(clear), .start(hh_start),
        .syn_current(hh_current), .done(hh_done), .spike(hh_spike),
        .membrane(hh_membrane)
    );

    neuron_array #(
        .NUM_NEURONS(NUM_NEURONS), .MEM_WIDTH(24), .CURRENT_WIDTH(24),
        .NEURON_MODEL(0), .EXECUTOR_COUNT(1)
    ) u_lif_array (
        .clk(clk), .rst(rst), .clear(clear), .update_valid(lif_array_valid),
        .syn_current(lif_array_current), .spike(lif_array_spike),
        .membrane(lif_array_membrane), .done(lif_array_done)
    );

    neuron_array #(
        .NUM_NEURONS(NUM_NEURONS), .MEM_WIDTH(24), .CURRENT_WIDTH(24),
        .NEURON_MODEL(1), .EXECUTOR_COUNT(1)
    ) u_izh_array (
        .clk(clk), .rst(rst), .clear(clear), .update_valid(izh_array_valid),
        .syn_current(izh_array_current), .spike(izh_array_spike),
        .membrane(izh_array_membrane), .done(izh_array_done)
    );

    neuron_array #(
        .NUM_NEURONS(NUM_NEURONS), .MEM_WIDTH(24), .CURRENT_WIDTH(24),
        .NEURON_MODEL(2), .EXECUTOR_COUNT(1)
    ) u_hh_array (
        .clk(clk), .rst(rst), .clear(clear), .update_valid(hh_array_valid),
        .syn_current(hh_array_current), .spike(hh_array_spike),
        .membrane(hh_array_membrane), .done(hh_array_done)
    );

    function automatic logic executor_done(input integer model);
        case (model)
            0: executor_done = lif_done;
            1: executor_done = izh_done;
            default: executor_done = hh_done;
        endcase
    endfunction

    function automatic logic executor_spike(input integer model);
        case (model)
            0: executor_spike = lif_spike;
            1: executor_spike = izh_spike;
            default: executor_spike = hh_spike;
        endcase
    endfunction

    function automatic logic array_done(input integer model);
        case (model)
            0: array_done = lif_array_done;
            1: array_done = izh_array_done;
            default: array_done = hh_array_done;
        endcase
    endfunction

    task automatic measure_executor(
        input integer model,
        input integer current_value,
        output integer cycles,
        output logic spike_seen
    );
        longint unsigned accepted_cycle;
        begin
            @(negedge clk);
            case (model)
                0: begin lif_current = current_value; lif_start = 1'b1; end
                1: begin izh_current = current_value; izh_start = 1'b1; end
                default: begin hh_current = current_value; hh_start = 1'b1; end
            endcase
            @(posedge clk);
            #1;
            accepted_cycle = cycle_counter;
            spike_seen = executor_spike(model);
            @(negedge clk);
            case (model)
                0: lif_start = 1'b0;
                1: izh_start = 1'b0;
                default: hh_start = 1'b0;
            endcase
            while (!executor_done(model)) begin
                @(posedge clk);
                #1;
                spike_seen |= executor_spike(model);
            end
            cycles = cycle_counter - accepted_cycle;
        end
    endtask

    task automatic measure_array(
        input integer model,
        output integer cycles
    );
        longint unsigned accepted_cycle;
        begin
            @(negedge clk);
            case (model)
                0: lif_array_valid = 1'b1;
                1: izh_array_valid = 1'b1;
                default: hh_array_valid = 1'b1;
            endcase
            @(posedge clk);
            #1;
            accepted_cycle = cycle_counter;
            @(negedge clk);
            case (model)
                0: lif_array_valid = 1'b0;
                1: izh_array_valid = 1'b0;
                default: hh_array_valid = 1'b0;
            endcase
            while (!array_done(model)) begin
                @(posedge clk);
                #1;
            end
            cycles = cycle_counter - accepted_cycle;
        end
    endtask

    initial begin
        integer lif_nonspike_cycles, lif_spike_cycles;
        integer izh_nonspike_cycles, izh_spike_cycles, ignored_cycles;
        integer hh_nonspike_cycles, hh_spike_cycles;
        integer lif_population_cycles, izh_population_cycles;
        integer hh_population_cycles;
        logic observed_spike;

        for (integer i = 0; i < NUM_NEURONS; i = i + 1) begin
            lif_array_current[i] = 0;
            izh_array_current[i] = 0;
            hh_array_current[i] = 0;
        end

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        measure_executor(0, 0, lif_nonspike_cycles, observed_spike);
        if (observed_spike)
            $fatal(1, "LIF non-spike case emitted a spike");
        measure_executor(0, 100, lif_spike_cycles, observed_spike);
        if (!observed_spike)
            $fatal(1, "LIF spike case did not emit a spike");

        measure_executor(1, 10, izh_nonspike_cycles, observed_spike);
        if (observed_spike)
            $fatal(1, "IZH first update unexpectedly spiked");
        for (integer step = 1; step < 3; step = step + 1)
            measure_executor(1, 10, ignored_cycles, observed_spike);
        measure_executor(1, 10, izh_spike_cycles, observed_spike);
        if (!observed_spike)
            $fatal(1, "IZH fourth update did not emit the reference spike");

        measure_executor(2, 0, hh_nonspike_cycles, observed_spike);
        if (observed_spike)
            $fatal(1, "HH first update unexpectedly spiked");
        for (integer step = 1; step < 5; step = step + 1)
            measure_executor(2, 0, ignored_cycles, observed_spike);
        measure_executor(2, 10, ignored_cycles, observed_spike);
        measure_executor(2, 10, hh_spike_cycles, observed_spike);
        if (!observed_spike)
            $fatal(1, "HH seventh update did not emit the reference spike");

        measure_array(0, lif_population_cycles);
        measure_array(1, izh_population_cycles);
        measure_array(2, hh_population_cycles);

        if (lif_nonspike_cycles != lif_spike_cycles
                || izh_nonspike_cycles != izh_spike_cycles
                || hh_nonspike_cycles != hh_spike_cycles)
            $fatal(1, "spike-dependent executor latency detected");
        if (lif_population_cycles != NUM_NEURONS * (lif_nonspike_cycles + 3)
                || izh_population_cycles != NUM_NEURONS * (izh_nonspike_cycles + 3)
                || hh_population_cycles != NUM_NEURONS * (hh_nonspike_cycles + 3))
            $fatal(1, "serialized scheduler cost is not three cycles per neuron");

        $display("UPDATE_COST model=LIF case=nonspike executor_cycles=%0d", lif_nonspike_cycles);
        $display("UPDATE_COST model=LIF case=spike executor_cycles=%0d", lif_spike_cycles);
        $display("UPDATE_COST model=IZH case=nonspike executor_cycles=%0d", izh_nonspike_cycles);
        $display("UPDATE_COST model=IZH case=spike executor_cycles=%0d", izh_spike_cycles);
        $display("UPDATE_COST model=HH case=nonspike executor_cycles=%0d", hh_nonspike_cycles);
        $display("UPDATE_COST model=HH case=spike executor_cycles=%0d", hh_spike_cycles);
        $display("SCHEDULER_COST model=LIF neurons=%0d population_cycles=%0d overhead_cycles=%0d",
                 NUM_NEURONS, lif_population_cycles,
                 lif_population_cycles - NUM_NEURONS * lif_nonspike_cycles);
        $display("SCHEDULER_COST model=IZH neurons=%0d population_cycles=%0d overhead_cycles=%0d",
                 NUM_NEURONS, izh_population_cycles,
                 izh_population_cycles - NUM_NEURONS * izh_nonspike_cycles);
        $display("SCHEDULER_COST model=HH neurons=%0d population_cycles=%0d overhead_cycles=%0d",
                 NUM_NEURONS, hh_population_cycles,
                 hh_population_cycles - NUM_NEURONS * hh_nonspike_cycles);
        $display("RESULT: PASS");
        $finish;
    end

    initial begin
        #10000000;
        $fatal(1, "timeout");
    end
endmodule
