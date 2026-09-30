`timescale 1ns/1ps

module tb_hodgkin_huxley_snn_top;
    logic clk = 1'b0;
    logic rst = 1'b1;
    logic start_inference = 1'b0;
    logic [2:0] input_event_count = 4;
    logic [7:0] timestep_count = 1;
    logic weight_write_valid = 1'b0;
    logic weight_layer = 1'b0;
    logic [1:0] weight_pre_idx = '0;
    logic weight_post_idx = 1'b0;
    logic signed [7:0] weight_write_data = '0;
    logic weight_write_ready;
    logic spike_write_valid = 1'b0;
    logic [1:0] spike_write_addr = '0;
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
    integer tx_count = 0;

    snn_top #(
        .NUM_EVENTS(4),
        .INPUT_NEURONS(4),
        .HIDDEN_NEURONS(1),
        .OUTPUT_NEURONS(1),
        .TIME_WIDTH(8),
        .MEM_WIDTH(24),
        .CURRENT_WIDTH(24),
        .WEIGHT_WIDTH(8),
        .COUNT_WIDTH(8),
        .HIDDEN_NEURON_MODEL(2),
        .OUTPUT_NEURON_MODEL(2)
    ) dut (.*);

    always #5 clk = ~clk;

    always @(posedge clk) begin
        #1;
        if (rst)
            tx_count = 0;
        else if (tx_valid) begin
            if (tx_data !== 1'b0)
                $fatal(1, "single output class must be zero");
            tx_count = tx_count + 1;
        end
    end

    task automatic write_weight(
        input logic layer_id,
        input logic [1:0] pre_index,
        input logic signed [7:0] value
    );
        begin
            wait (weight_write_ready);
            @(negedge clk);
            weight_layer = layer_id;
            weight_pre_idx = pre_index;
            weight_post_idx = 1'b0;
            weight_write_data = value;
            weight_write_valid = 1'b1;
            @(negedge clk);
            weight_write_valid = 1'b0;
        end
    endtask

    task automatic write_event(input logic [1:0] index);
        begin
            wait (spike_write_ready);
            @(negedge clk);
            spike_write_addr = index;
            spike_write_timestamp = 8'd0;
            spike_write_neuron_id = index;
            spike_write_valid = 1'b1;
            @(negedge clk);
            spike_write_valid = 1'b0;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        rst = 1'b0;

        for (integer input_index = 0; input_index < 4; input_index++) begin
            write_weight(0, input_index[1:0], 8'sd127);
            write_event(input_index[1:0]);
        end
        write_weight(1, 0, 8'sd127);

        @(negedge clk);
        start_inference = 1'b1;
        @(negedge clk);
        start_inference = 1'b0;

        wait (inference_done);
        @(posedge clk);
        #2;
        if (dut.class_count[0] !== 1)
            $fatal(1, "class count=%0d expected 1", dut.class_count[0]);
        if (tx_count !== 1)
            $fatal(1, "prediction transactions=%0d expected 1", tx_count);
        if ($signed(dut.layer0.u_neuron_array.membrane[0]) !== 24'sd399797)
            $fatal(1, "hidden voltage=%0d expected 399797",
                $signed(dut.layer0.u_neuron_array.membrane[0]));
        if ($signed(dut.layer1.u_neuron_array.membrane[0]) !== 24'sd414632)
            $fatal(1, "output voltage=%0d expected 414632",
                $signed(dut.layer1.u_neuron_array.membrane[0]));
        if (synapse_phase_cycles + hidden_update_phase_cycles +
            hidden_routing_phase_cycles + output_update_phase_cycles +
            control_overhead_phase_cycles !== inference_cycles)
            $fatal(1, "phase cycles do not sum to inference cycles");

        $display("RESULT: PASS (bit-exact two-layer Hodgkin-Huxley inference)");
        $finish;
    end

    initial begin
        #300000;
        $fatal(1, "timeout");
    end
endmodule
