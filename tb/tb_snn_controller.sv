`timescale 1ns/1ps

module tb_snn_controller;
    logic clk = 1'b0;
    logic rst = 1'b1;
    logic start_inference = 1'b0;
    logic event_valid = 1'b0;
    logic [7:0] event_timestamp = '0;
    logic end_of_stream = 1'b0;
    logic [7:0] timestep_count = 1;
    logic input_event_ready = 1'b1;
    logic layer0_done = 1'b0;
    logic encoder_done = 1'b0;
    logic layer1_ready = 1'b1;
    logic layer1_done = 1'b0;
    logic output_done = 1'b0;
    logic read_event, route_event, update_layer0, start_encoder;
    logic update_layer1, start_output_decode, inference_done;
    logic controller_idle;
    logic [2:0] telemetry_phase;

    snn_controller #(.TIME_WIDTH(8)) dut (.*);
    always #5 clk = ~clk;

    task automatic check_high(input string name, input logic actual);
        if (actual !== 1'b1) $fatal(1, "%s was not asserted", name);
    endtask

    initial begin
        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(negedge clk); start_inference = 1'b1;
        @(negedge clk); start_inference = 1'b0;

        event_valid = 1'b1;
        event_timestamp = 8'd0;
        end_of_stream = 1'b1;
        #1;
        check_high("read_event", read_event);
        check_high("route_event", route_event);
        @(posedge clk);
        #1 event_valid = 1'b0;

        wait (update_layer0);
        @(posedge clk);
        @(negedge clk); layer0_done = 1'b1;
        @(negedge clk); layer0_done = 1'b0;
        wait (start_encoder);
        @(posedge clk);
        @(negedge clk); encoder_done = 1'b1;
        @(negedge clk); encoder_done = 1'b0;
        wait (update_layer1);
        @(posedge clk);
        @(negedge clk); layer1_done = 1'b1;
        @(negedge clk); layer1_done = 1'b0;
        wait (start_output_decode);
        @(posedge clk);
        @(negedge clk); output_done = 1'b1;
        @(negedge clk); output_done = 1'b0;
        wait (inference_done);

        if (dut.current_time !== 8'd1)
            $fatal(1, "current_time=%0d, expected 1", dut.current_time);
        $display("RESULT: PASS");
        $finish;
    end

    initial begin
        #2000;
        $fatal(1, "timeout");
    end
endmodule
