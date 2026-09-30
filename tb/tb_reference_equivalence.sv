`timescale 1ns/1ps

module tb_reference_equivalence #(
    parameter bit SERIAL_LIF = 1'b0
);
    localparam int INPUT_NEURONS = 4;
    localparam int HIDDEN_NEURONS = 3;
    localparam int OUTPUT_NEURONS = 2;
    localparam int NUM_EVENTS = 8;
    localparam int MEM_WIDTH = 16;
    localparam int COUNT_WIDTH = 8;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic start_inference = 1'b0;
    logic [3:0] input_event_count = NUM_EVENTS;
    logic [7:0] timestep_count = 1;
    logic weight_write_valid = 1'b0;
    logic weight_layer = 1'b0;
    logic [1:0] weight_pre_idx = '0;
    logic [1:0] weight_post_idx = '0;
    logic signed [7:0] weight_write_data = '0;
    logic weight_write_ready;
    logic spike_write_valid = 1'b0;
    logic [2:0] spike_write_addr = '0;
    logic [7:0] spike_write_timestamp = '0;
    logic [1:0] spike_write_neuron_id = '0;
    logic spike_write_ready;
    logic inference_done;
    logic tx_valid;
    logic tx_data;
    logic [63:0] inference_cycles;
    logic [31:0] consumed_input_events;
    logic [31:0] emitted_hidden_spikes;
    logic [31:0] emitted_output_spikes;
    logic [63:0] synapse_phase_cycles;
    logic [63:0] hidden_update_phase_cycles;
    logic [63:0] hidden_routing_phase_cycles;
    logic [63:0] output_update_phase_cycles;
    logic [63:0] control_overhead_phase_cycles;

    logic signed [7:0] weights_l0 [0:INPUT_NEURONS*HIDDEN_NEURONS-1];
    logic signed [7:0] weights_l1 [0:HIDDEN_NEURONS*OUTPUT_NEURONS-1];
    logic [7:0] event_times [0:NUM_EVENTS-1];
    logic [7:0] event_ids [0:NUM_EVENTS-1];
    logic signed [MEM_WIDTH-1:0] expected_hidden [0:HIDDEN_NEURONS-1];
    logic signed [MEM_WIDTH-1:0] expected_output [0:OUTPUT_NEURONS-1];
    logic [COUNT_WIDTH-1:0] expected_counts [0:OUTPUT_NEURONS-1];
    logic expected_prediction_valid [0:0];
    logic [7:0] expected_prediction [0:0];

    integer tx_count = 0;
    logic captured_prediction = 1'b0;
    logic previous_tx_valid = 1'b0;
    string case_name;

    snn_top #(
        .NUM_EVENTS(NUM_EVENTS),
        .INPUT_NEURONS(INPUT_NEURONS),
        .HIDDEN_NEURONS(HIDDEN_NEURONS),
        .OUTPUT_NEURONS(OUTPUT_NEURONS),
        .TIME_WIDTH(8),
        .MEM_WIDTH(MEM_WIDTH),
        .CURRENT_WIDTH(MEM_WIDTH),
        .WEIGHT_WIDTH(8),
        .COUNT_WIDTH(COUNT_WIDTH),
        .HIDDEN_EXECUTOR_COUNT(SERIAL_LIF ? 1 : HIDDEN_NEURONS),
        .OUTPUT_EXECUTOR_COUNT(SERIAL_LIF ? 1 : OUTPUT_NEURONS),
        .THRESHOLD(50),
        .RESET_POT(0),
        .LEAK_SHIFT(2)
    ) dut (.*);

    always #5 clk = ~clk;

    task automatic write_weight(
        input logic layer_id,
        input logic [1:0] pre,
        input logic [1:0] post,
        input logic signed [7:0] value
    );
        wait (weight_write_ready);
        @(negedge clk);
        weight_layer = layer_id;
        weight_pre_idx = pre;
        weight_post_idx = post;
        weight_write_data = value;
        weight_write_valid = 1'b1;
        @(negedge clk);
        weight_write_valid = 1'b0;
    endtask

    task automatic write_event(input int index);
        wait (spike_write_ready);
        @(negedge clk);
        spike_write_addr = index[2:0];
        spike_write_timestamp = event_times[index];
        spike_write_neuron_id = event_ids[index][1:0];
        spike_write_valid = 1'b1;
        @(negedge clk);
        spike_write_valid = 1'b0;
    endtask

    // Public-interface invariants checked throughout configuration/inference.
    always @(posedge clk) begin
        #1;
        if (rst) begin
            tx_count = 0;
            captured_prediction = 1'b0;
            previous_tx_valid = 1'b0;
        end
        else begin
            if ((weight_write_ready || spike_write_ready) && !dut.controller_idle)
                $fatal(1, "configuration ready asserted while controller was active");
            if (tx_valid && previous_tx_valid)
                $fatal(1, "tx_valid must be a one-cycle pulse");
            if (tx_valid) begin
                tx_count = tx_count + 1;
                captured_prediction = tx_data;
            end
            previous_tx_valid = tx_valid;
        end
    end

    initial begin
        $readmemh("sim/equivalence/weights_l0.mem", weights_l0);
        $readmemh("sim/equivalence/weights_l1.mem", weights_l1);
        $readmemh("sim/equivalence/event_times.mem", event_times);
        timestep_count = event_times[NUM_EVENTS-1] + 1'b1;
        $readmemh("sim/equivalence/event_ids.mem", event_ids);
        $readmemh("sim/equivalence/expected_hidden.mem", expected_hidden);
        $readmemh("sim/equivalence/expected_output.mem", expected_output);
        $readmemh("sim/equivalence/expected_counts.mem", expected_counts);
        $readmemh("sim/equivalence/expected_prediction_valid.mem", expected_prediction_valid);
        $readmemh("sim/equivalence/expected_prediction.mem", expected_prediction);
        if (!$value$plusargs("CASE=%s", case_name))
            case_name = "unnamed";

        repeat (3) @(posedge clk);
        rst = 1'b0;

        for (int pre = 0; pre < INPUT_NEURONS; pre++)
            for (int post = 0; post < HIDDEN_NEURONS; post++)
                write_weight(0, pre[1:0], post[1:0], weights_l0[pre*HIDDEN_NEURONS+post]);
        for (int pre = 0; pre < HIDDEN_NEURONS; pre++)
            for (int post = 0; post < OUTPUT_NEURONS; post++)
                write_weight(1, pre[1:0], post[1:0], weights_l1[pre*OUTPUT_NEURONS+post]);
        for (int event_index = 0; event_index < NUM_EVENTS; event_index++)
            write_event(event_index);

        @(negedge clk);
        start_inference = 1'b1;
        @(negedge clk);
        start_inference = 1'b0;

        // Requests may be presented while active, but ready must remain low.
        weight_write_valid = 1'b1;
        spike_write_valid = 1'b1;
        @(posedge clk);
        #1;
        if (weight_write_ready || spike_write_ready)
            $fatal(1, "configuration request accepted during inference");
        @(negedge clk);
        weight_write_valid = 1'b0;
        spike_write_valid = 1'b0;

        wait (inference_done);
        @(posedge clk);
        #2;

        for (int index = 0; index < HIDDEN_NEURONS; index++)
            if (dut.layer0.u_neuron_array.membrane[index] !== expected_hidden[index])
                $fatal(1, "hidden[%0d]=%0d expected %0d", index,
                       dut.layer0.u_neuron_array.membrane[index], expected_hidden[index]);
        for (int index = 0; index < OUTPUT_NEURONS; index++) begin
            if (dut.layer1.u_neuron_array.membrane[index] !== expected_output[index])
                $fatal(1, "output[%0d]=%0d expected %0d", index,
                       dut.layer1.u_neuron_array.membrane[index], expected_output[index]);
            if (dut.class_count[index] !== expected_counts[index])
                $fatal(1, "count[%0d]=%0d expected %0d", index,
                       dut.class_count[index], expected_counts[index]);
        end

        if (expected_prediction_valid[0]) begin
            if (tx_count !== 1)
                $fatal(1, "expected one prediction transaction, got %0d", tx_count);
            if (captured_prediction !== expected_prediction[0][0])
                $fatal(1, "prediction=%0d expected %0d", captured_prediction,
                       expected_prediction[0][0]);
        end
        else if (tx_count !== 0) begin
            $fatal(1, "unexpected prediction transaction");
        end
        if (synapse_phase_cycles + hidden_update_phase_cycles +
            hidden_routing_phase_cycles + output_update_phase_cycles +
            control_overhead_phase_cycles !== inference_cycles)
            $fatal(1, "phase cycles do not sum to inference cycles");

        $display("RESULT: PASS (%s)", case_name);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "timeout");
    end
endmodule
