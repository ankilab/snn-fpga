`timescale 1ns/1ps

module tb_spike_event_encoder;
    localparam int NUM_NEURONS = 8;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic [NUM_NEURONS-1:0] spikes = '0;
    logic start = 1'b0;
    logic event_ready = 1'b0;
    logic event_valid;
    logic [$clog2(NUM_NEURONS)-1:0] neuron_id;
    logic done;
    int seen = 0;

    spike_event_encoder #(.NUM_NEURONS(NUM_NEURONS)) dut (.*);
    always #5 clk = ~clk;

    task automatic fail(input string message);
        $error("%s", message);
        $fatal(1);
    endtask

    initial begin
        repeat (2) @(posedge clk);
        rst <= 1'b0;
        spikes <= 8'b00101000; // neuron 3 and 5
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        wait (event_valid);
        if (neuron_id !== 3) fail("first event must be neuron 3");

        // Backpressure must keep both valid and data stable.
        repeat (3) begin
            @(posedge clk);
            #1;
            if (!event_valid || neuron_id !== 3)
                fail("event changed while event_ready was low");
        end

        event_ready <= 1'b1;
        @(posedge clk);
        event_ready <= 1'b0;

        wait (!event_valid);
        wait (event_valid);
        if (neuron_id !== 5) fail("second event must be neuron 5");
        event_ready <= 1'b1;
        @(posedge clk);
        event_ready <= 1'b0;

        wait (!event_valid);
        wait (done);
        $display("RESULT: PASS");
        $finish;
    end

    initial begin
        #2000;
        fail("timeout");
    end
endmodule
