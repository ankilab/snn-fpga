`timescale 1ns/1ps
//------------------------------------------------------------------------------
// Testbench: spike_reader_tb
//
// Self-checking testbench for spike_reader. Preloads spike_memory with known
// pattern data via hierarchical reference (spike_memory has no write port
// exposed through spike_reader, since writes are tied off during inference),
// then drives next_event and checks:
//   1. event_valid behaves correctly around reset
//   2. timestamp / neuron_id match the expected preloaded pattern
//   3. end_of_stream pulses exactly once, on the correct cycle
//   4. event_valid does NOT spuriously reassert after end_of_stream
//      (regression test for the "!event_valid_reg re-arm" bug)
//
// Waveforms are dumped to spike_reader_tb.vcd for viewing in GTKWave.
//------------------------------------------------------------------------------

module spike_reader_tb;

    // Small event count so the waveform stays easy to read in GTKWave.
    localparam int NUM_EVENTS      = 8;
    localparam int TIME_WIDTH      = 32;
    localparam int NEURON_ID_WIDTH = 8;
    localparam int ADDR_WIDTH      = $clog2(NUM_EVENTS);

    localparam time CLK_PERIOD = 10ns;

    // DUT I/O
    logic clk;
    logic rst;
    logic next_event;
    logic [$clog2(NUM_EVENTS+1)-1:0] event_count = NUM_EVENTS;

    logic event_valid;
    logic [TIME_WIDTH-1:0]      timestamp;
    logic [NEURON_ID_WIDTH-1:0] neuron_id;
    logic end_of_stream;

    int unsigned error_count;
    int unsigned checks_run;

    // Expected pattern: timestamp = 100*i, neuron_id = i (mod 256)
    function automatic logic [TIME_WIDTH-1:0] exp_time(int i);
        return TIME_WIDTH'(i * 100);
    endfunction

    function automatic logic [NEURON_ID_WIDTH-1:0] exp_neuron(int i);
        return NEURON_ID_WIDTH'(i);
    endfunction

    //--------------------------------------------------------------------
    // DUT instantiation
    //--------------------------------------------------------------------
    spike_reader #(
        .NUM_EVENTS     (NUM_EVENTS),
        .TIME_WIDTH     (TIME_WIDTH),
        .NEURON_ID_WIDTH(NEURON_ID_WIDTH)
    ) dut (
        .clk          (clk),
        .rst          (rst),
        .next_event   (next_event),
        .restart_stream(1'b0),
        .event_count   (event_count),
        .config_write (1'b0),
        .config_addr  ('0),
        .config_timestamp('0),
        .config_neuron_id('0),
        .event_valid  (event_valid),
        .timestamp    (timestamp),
        .neuron_id    (neuron_id),
        .end_of_stream(end_of_stream)
    );

    //--------------------------------------------------------------------
    // Clock generation
    //--------------------------------------------------------------------
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //--------------------------------------------------------------------
    // Memory preload (hierarchical force, since spike_reader ties
    // write_en to 0 permanently -- there is no other way to get known
    // data into spike_memory from outside during inference mode).
    //--------------------------------------------------------------------
    task automatic preload_memory();
        for (int i = 0; i < NUM_EVENTS; i++) begin
            dut.mem_i.time_mem[i]   = exp_time(i);
            dut.mem_i.neuron_mem[i] = exp_neuron(i);
        end
    endtask

    //--------------------------------------------------------------------
    // Convenience: advance one clock edge
    //--------------------------------------------------------------------
    task automatic tick();
        @(posedge clk);
    endtask

    //--------------------------------------------------------------------
    // Checking helpers
    //--------------------------------------------------------------------
    task automatic check_equal(string name, logic [63:0] got, logic [63:0] exp);
        checks_run++;
        if (got !== exp) begin
            error_count++;
            $error("[%0t] MISMATCH %s: got=0x%0h expected=0x%0h", $time, name, got, exp);
        end
    endtask

    task automatic check_true(string name, logic got);
        checks_run++;
        if (got !== 1'b1) begin
            error_count++;
            $error("[%0t] EXPECTED TRUE: %s", $time, name);
        end
    endtask

    task automatic check_false(string name, logic got);
        checks_run++;
        if (got !== 1'b0) begin
            error_count++;
            $error("[%0t] EXPECTED FALSE (but was 1): %s", $time, name);
        end
    endtask

    //--------------------------------------------------------------------
    // Main stimulus
    //--------------------------------------------------------------------
    initial begin
        // VCD dump for GTKWave
        $dumpfile("./sim/sim_spike_reader.vcd");
        $dumpvars(0, spike_reader_tb);

        error_count = 0;
        checks_run  = 0;
        rst         = 1'b1;
        next_event  = 1'b0;

        preload_memory();

        // Hold reset for a few cycles
        repeat (3) tick();

        check_false("event_valid during reset", event_valid);
        check_false("end_of_stream during reset", end_of_stream);

        // Release reset
        rst = 1'b0;
        tick(); // event_ptr <= 0, event_valid_reg <= 1 takes effect on this edge
        #1; // let combinational outputs settle after the edge

        check_true ("event_valid after reset release", event_valid);
        check_equal("timestamp[0]", timestamp, exp_time(0));
        check_equal("neuron_id[0]", neuron_id, exp_neuron(0));
        check_false("end_of_stream at event 0", end_of_stream);

        // Walk through events 1 .. NUM_EVENTS-1 by pulsing next_event.
        // Each pulse advances event_ptr by one and PRESENTS the next
        // event; it does not yet retire the event it moves away from.
        for (int i = 1; i < NUM_EVENTS; i++) begin
            next_event = 1'b1;
            tick();
            #1;

            if (i < NUM_EVENTS - 1) begin
                check_false($sformatf("end_of_stream at event %0d", i), end_of_stream);
            end
            else begin
                // i == NUM_EVENTS-1: event_ptr has just reached
                // LAST_EVENT and next_event is still held -> end_of_stream
                // is combinationally asserted, signalling "the event
                // currently presented is the last one". The event itself
                // is not yet retired (event_valid_reg is still 1) until a
                // FURTHER next_event pulse arrives while parked here.
                check_true($sformatf("end_of_stream at last event %0d", i), end_of_stream);
            end

            check_true ($sformatf("event_valid at event %0d", i), event_valid);
            check_equal($sformatf("timestamp[%0d]", i), timestamp, exp_time(i));
            check_equal($sformatf("neuron_id[%0d]", i), neuron_id, exp_neuron(i));

            next_event = 1'b0;
        end

        // Retiring the last event requires one MORE next_event pulse
        // while event_ptr is already parked at LAST_EVENT. This is the
        // pulse that actually clears event_valid_reg and latches
        // stream_done. In total, fully draining NUM_EVENTS events takes
        // NUM_EVENTS next_event pulses, not NUM_EVENTS-1.
        next_event = 1'b1;
        tick();
        #1;
        next_event = 1'b0;

        check_false("event_valid immediately after last event is retired", event_valid);
        check_false("end_of_stream deasserts once the stream is retired", end_of_stream);

        // Regression check for the previously reported bug: without any
        // next_event and without reset, event_valid must NOT come back
        // on its own. An earlier version of this RTL re-armed
        // event_valid_reg here because it only tested "!event_valid_reg",
        // which was also true right after the stream ended (not just
        // right after reset). The stream_done latch is what should now
        // prevent that.
        tick();
        #1;
        check_false("event_valid must stay low after stream end (no spurious re-arm)", event_valid);

        tick();
        #1;
        check_false("event_valid still low two cycles after stream end", event_valid);

        // Sanity: pulsing next_event post-stream must not do anything
        // strange (pointer should not move / valid should not assert).
        next_event = 1'b1;
        tick();
        #1;
        check_false("event_valid stays low even if next_event pulses post-stream", event_valid);
        next_event = 1'b0;

        // Now verify a mid-stream reset correctly restarts the stream.
        rst = 1'b1;
        tick();
        rst = 1'b0;
        tick();
        #1;
        check_true ("event_valid after mid-stream reset", event_valid);
        check_equal("timestamp back to event 0 after reset", timestamp, exp_time(0));

        //----------------------------------------------------------------
        // Report
        //----------------------------------------------------------------
        tick();
        $display("--------------------------------------------------");
        $display("Checks run   : %0d", checks_run);
        $display("Errors found : %0d", error_count);
        if (error_count == 0)
            $display("RESULT: PASS");
        else
            $display("RESULT: FAIL");
        $display("--------------------------------------------------");

        $finish;
    end

    // Safety timeout
    initial begin
        #100000;
        $display("ERROR: Testbench timeout - simulation did not finish");
        $finish;
    end

endmodule
