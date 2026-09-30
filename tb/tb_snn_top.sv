`timescale 1ns/1ps

module tb_snn_top;
    localparam int INPUT_NEURONS  = 4;
    localparam int HIDDEN_NEURONS = 3;
    localparam int OUTPUT_NEURONS = 2;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic start_inference = 1'b0;
    logic [1:0] input_event_count = 2;
    logic [7:0] timestep_count = 1;
    logic weight_write_valid = 1'b0;
    logic weight_layer = 1'b0;
    logic [1:0] weight_pre_idx = '0;
    logic [1:0] weight_post_idx = '0;
    logic signed [7:0] weight_write_data = '0;
    logic weight_write_ready;
    logic spike_write_valid = 1'b0;
    logic spike_write_addr = 1'b0;
    logic [7:0] spike_write_timestamp = '0;
    logic [1:0] spike_write_neuron_id = '0;
    logic spike_write_ready;
    logic inference_done;
    logic tx_valid;
    logic [$clog2(OUTPUT_NEURONS)-1:0] tx_data;
    logic [63:0] inference_cycles;
    logic [31:0] consumed_input_events;
    logic [31:0] emitted_hidden_spikes;
    logic [31:0] emitted_output_spikes;
    logic [63:0] synapse_phase_cycles;
    logic [63:0] hidden_update_phase_cycles;
    logic [63:0] hidden_routing_phase_cycles;
    logic [63:0] output_update_phase_cycles;
    logic [63:0] control_overhead_phase_cycles;
    logic prediction_seen = 1'b0;
    int unsigned tx_count = 0;
    logic [$clog2(OUTPUT_NEURONS)-1:0] captured_prediction;

    snn_top #(
        .NUM_EVENTS(2),
        .INPUT_NEURONS(INPUT_NEURONS),
        .HIDDEN_NEURONS(HIDDEN_NEURONS),
        .OUTPUT_NEURONS(OUTPUT_NEURONS),
        .TIME_WIDTH(8),
        .MEM_WIDTH(16),
        .CURRENT_WIDTH(16),
        .WEIGHT_WIDTH(8),
        .COUNT_WIDTH(8),
        .THRESHOLD(100),
        .RESET_POT(0),
        .LEAK_SHIFT(2)
    ) dut (.*);

    always #5 clk = ~clk;

    task automatic write_weight(input logic layer, input logic [1:0] pre,
                                input logic [1:0] post,
                                input logic signed [7:0] value);
        wait (weight_write_ready);
        @(negedge clk);
        weight_layer = layer;
        weight_pre_idx = pre;
        weight_post_idx = post;
        weight_write_data = value;
        weight_write_valid = 1'b1;
        @(negedge clk);
        weight_write_valid = 1'b0;
    endtask

    task automatic write_spike(input logic addr, input logic [7:0] timestamp,
                               input logic [1:0] neuron_id);
        wait (spike_write_ready);
        @(negedge clk);
        spike_write_addr = addr;
        spike_write_timestamp = timestamp;
        spike_write_neuron_id = neuron_id;
        spike_write_valid = 1'b1;
        @(negedge clk);
        spike_write_valid = 1'b0;
    endtask

    always @(posedge clk) begin
        #1;
        if (rst) begin
            prediction_seen = 1'b0;
            captured_prediction = '0;
            tx_count = 0;
        end
        else if (tx_valid) begin
            prediction_seen = 1'b1;
            captured_prediction = tx_data;
            tx_count = tx_count + 1;
        end
    end

    initial begin
        repeat (3) @(posedge clk);
        rst = 1'b0;

        // Two input spikes at t=0: input 0 and input 1.
        write_spike(0, 8'd0, 2'd0);
        write_spike(1, 8'd0, 2'd1);

        // Initialize every weight through the public configuration port.
        for (int pre = 0; pre < INPUT_NEURONS; pre++)
            for (int post = 0; post < HIDDEN_NEURONS; post++)
                write_weight(0, pre[1:0], post[1:0], 8'sd0);

        for (int pre = 0; pre < HIDDEN_NEURONS; pre++)
            for (int post = 0; post < OUTPUT_NEURONS; post++)
                write_weight(1, pre[1:0], post[1:0], 8'sd0);

        // 60 + 60 makes hidden neuron 0 spike; it drives output class 1.
        write_weight(0, 0, 0, 8'sd60);
        write_weight(0, 1, 0, 8'sd60);
        // Hidden neuron 1 intentionally retains a subthreshold potential.
        write_weight(0, 0, 1, 8'sd40);
        write_weight(1, 0, 1, 8'sd120);

        @(negedge clk);
        start_inference = 1'b1;
        @(negedge clk);
        start_inference = 1'b0;
        #1;
        if (weight_write_ready)
            $fatal(1, "weight writes must be blocked during inference");
        if (spike_write_ready)
            $fatal(1, "spike writes must be blocked during inference");

        wait (inference_done);
        @(posedge clk);
        #1;

        if (!prediction_seen)
            $fatal(1, "no prediction was transmitted");
        if (captured_prediction !== 1)
            $fatal(1, "prediction=%0d, expected class 1", captured_prediction);
        if (tx_count !== 1)
            $fatal(1, "tx_valid was asserted %0d times, expected once", tx_count);
        if (dut.class_count[0] !== 0 || dut.class_count[1] !== 1)
            $fatal(1, "class counts=[%0d,%0d], expected [0,1]",
                   dut.class_count[0], dut.class_count[1]);
        if (dut.layer0.u_neuron_array.membrane[1] !== 16'sd40)
            $fatal(1, "first inference hidden membrane=%0d, expected 40",
                   dut.layer0.u_neuron_array.membrane[1]);

        // Run the same configured network again without asserting global reset.
        wait (spike_write_ready);
        @(negedge clk);
        prediction_seen = 1'b0;
        tx_count = 0;
        start_inference = 1'b1;
        @(negedge clk);
        start_inference = 1'b0;

        wait (inference_done);
        @(posedge clk);
        #1;
        if (!prediction_seen || captured_prediction !== 1 || tx_count !== 1)
            $fatal(1, "second inference did not produce exactly one class-1 result");
        if (dut.class_count[0] !== 0 || dut.class_count[1] !== 1)
            $fatal(1, "class counters were not cleared for second inference");
        if (dut.layer0.u_neuron_array.membrane[1] !== 16'sd40)
            $fatal(1, "soft reset failed: second hidden membrane=%0d, expected 40",
                   dut.layer0.u_neuron_array.membrane[1]);
        if (synapse_phase_cycles + hidden_update_phase_cycles +
            hidden_routing_phase_cycles + output_update_phase_cycles +
            control_overhead_phase_cycles !== inference_cycles)
            $fatal(1, "phase cycles do not sum to inference cycles");

        $display("RESULT: PASS (two inferences, prediction=%0d)", captured_prediction);
        $finish;
    end

    initial begin
        #10000;
        $fatal(1, "timeout waiting for end-to-end inference");
    end
endmodule
