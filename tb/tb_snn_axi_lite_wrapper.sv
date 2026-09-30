`timescale 1ns/1ps

module tb_snn_axi_lite_wrapper;
    localparam logic [1:0] AXI_OKAY = 2'b00;
    localparam logic [1:0] AXI_SLVERR = 2'b10;
    localparam logic [1:0] AXI_DECERR = 2'b11;

    logic s_axi_aclk = 1'b0;
    logic s_axi_aresetn = 1'b0;
    logic [7:0] s_axi_awaddr = '0;
    logic [2:0] s_axi_awprot = '0;
    logic s_axi_awvalid = 1'b0;
    logic s_axi_awready;
    logic [31:0] s_axi_wdata = '0;
    logic [3:0] s_axi_wstrb = '0;
    logic s_axi_wvalid = 1'b0;
    logic s_axi_wready;
    logic [1:0] s_axi_bresp;
    logic s_axi_bvalid;
    logic s_axi_bready = 1'b0;
    logic [7:0] s_axi_araddr = '0;
    logic [2:0] s_axi_arprot = '0;
    logic s_axi_arvalid = 1'b0;
    logic s_axi_arready;
    logic [31:0] s_axi_rdata;
    logic [1:0] s_axi_rresp;
    logic s_axi_rvalid;
    logic s_axi_rready = 1'b0;
    logic irq;

    logic previous_b_stall = 1'b0;
    logic [1:0] stalled_bresp = '0;
    logic previous_r_stall = 1'b0;
    logic [31:0] stalled_rdata = '0;
    logic [1:0] stalled_rresp = '0;
    logic [31:0] ignored;
    logic [31:0] first_cycles;
    logic [31:0] replay_cycles;
    logic [63:0] synapse_cycles;
    logic [63:0] hidden_update_cycles;
    logic [63:0] hidden_routing_cycles;
    logic [63:0] output_update_cycles;
    logic [63:0] control_overhead_cycles;
    logic [31:0] phase_low_word;
    logic [31:0] phase_high_word;

    snn_axi_lite_wrapper #(
        .NUM_EVENTS(2),
        .INPUT_NEURONS(2),
        .HIDDEN_NEURONS(2),
        .OUTPUT_NEURONS(2),
        .TIME_WIDTH(8),
        .MEM_WIDTH(16),
        .CURRENT_WIDTH(16),
        .WEIGHT_WIDTH(8),
        .COUNT_WIDTH(8),
        .INTERFACE_VERSION(32'h0001_0006),
        .THRESHOLD(100),
        .LEAK_SHIFT(2)
    ) dut (.*);

    always #5 s_axi_aclk = ~s_axi_aclk;

    // Responses and payload must remain stable while the master applies
    // backpressure. These are core AXI4-Lite channel invariants.
    always @(posedge s_axi_aclk) begin
        #1;
        if (!s_axi_aresetn) begin
            previous_b_stall = 1'b0;
            previous_r_stall = 1'b0;
        end
        else begin
            if (previous_b_stall && !s_axi_bready &&
                (!s_axi_bvalid || s_axi_bresp !== stalled_bresp))
                $fatal(1, "B channel changed while BREADY was low");
            if (previous_r_stall && !s_axi_rready &&
                (!s_axi_rvalid || s_axi_rdata !== stalled_rdata ||
                 s_axi_rresp !== stalled_rresp))
                $fatal(1, "R channel changed while RREADY was low");

            previous_b_stall = s_axi_bvalid && !s_axi_bready;
            stalled_bresp = s_axi_bresp;
            previous_r_stall = s_axi_rvalid && !s_axi_rready;
            stalled_rdata = s_axi_rdata;
            stalled_rresp = s_axi_rresp;
        end
    end

    task automatic send_aw(input logic [7:0] address, input int delay_cycles);
        repeat (delay_cycles) @(posedge s_axi_aclk);
        @(negedge s_axi_aclk);
        s_axi_awaddr = address;
        s_axi_awvalid = 1'b1;
        do @(posedge s_axi_aclk); while (!s_axi_awready);
        @(negedge s_axi_aclk);
        s_axi_awvalid = 1'b0;
    endtask

    task automatic send_w(
        input logic [31:0] data,
        input logic [3:0] strobes,
        input int delay_cycles
    );
        repeat (delay_cycles) @(posedge s_axi_aclk);
        @(negedge s_axi_aclk);
        s_axi_wdata = data;
        s_axi_wstrb = strobes;
        s_axi_wvalid = 1'b1;
        do @(posedge s_axi_aclk); while (!s_axi_wready);
        @(negedge s_axi_aclk);
        s_axi_wvalid = 1'b0;
    endtask

    task automatic axi_write(
        input logic [7:0] address,
        input logic [31:0] data,
        input logic [3:0] strobes,
        input logic [1:0] expected_response,
        input int aw_delay,
        input int w_delay
    );
        fork
            send_aw(address, aw_delay);
            send_w(data, strobes, w_delay);
        join

        wait (s_axi_bvalid);
        repeat (2) begin
            @(posedge s_axi_aclk);
            #2;
            if (!s_axi_bvalid || s_axi_bresp !== expected_response)
                $fatal(1, "write 0x%02x response changed/incorrect: %0b",
                       address, s_axi_bresp);
        end
        @(negedge s_axi_aclk);
        s_axi_bready = 1'b1;
        @(posedge s_axi_aclk);
        @(negedge s_axi_aclk);
        s_axi_bready = 1'b0;
    endtask

    task automatic axi_read(
        input logic [7:0] address,
        input logic [1:0] expected_response,
        output logic [31:0] data
    );
        @(negedge s_axi_aclk);
        s_axi_araddr = address;
        s_axi_arvalid = 1'b1;
        do @(posedge s_axi_aclk); while (!s_axi_arready);
        @(negedge s_axi_aclk);
        s_axi_arvalid = 1'b0;

        wait (s_axi_rvalid);
        data = s_axi_rdata;
        repeat (2) begin
            @(posedge s_axi_aclk);
            #2;
            if (!s_axi_rvalid || s_axi_rresp !== expected_response ||
                s_axi_rdata !== data)
                $fatal(1, "read 0x%02x changed: valid=%0b resp=%0b/%0b data=%08x/%08x",
                    address, s_axi_rvalid, s_axi_rresp, expected_response,
                    s_axi_rdata, data);
        end
        @(negedge s_axi_aclk);
        s_axi_rready = 1'b1;
        @(posedge s_axi_aclk);
        @(negedge s_axi_aclk);
        s_axi_rready = 1'b0;
    endtask

    task automatic expect_read(
        input logic [7:0] address,
        input logic [31:0] expected
    );
        logic [31:0] actual;
        axi_read(address, AXI_OKAY, actual);
        if (actual !== expected)
            $fatal(1, "read 0x%02x=%08x expected %08x", address, actual, expected);
    endtask

    task automatic axi_read_phase(
        input logic [7:0] address,
        output logic [31:0] data
    );
        @(negedge s_axi_aclk);
        s_axi_araddr = address;
        s_axi_arvalid = 1'b1;
        do @(posedge s_axi_aclk); while (!s_axi_arready);
        @(negedge s_axi_aclk);
        s_axi_arvalid = 1'b0;
        wait (s_axi_rvalid);
        if (s_axi_rresp !== AXI_OKAY)
            $fatal(1, "phase read 0x%02x response=%0b", address, s_axi_rresp);
        data = s_axi_rdata;
        @(negedge s_axi_aclk);
        s_axi_rready = 1'b1;
        @(posedge s_axi_aclk);
        @(negedge s_axi_aclk);
        s_axi_rready = 1'b0;
    endtask

    initial begin
        repeat (4) @(posedge s_axi_aclk);
        s_axi_aresetn = 1'b1;

        expect_read(8'h00, 32'h534E4E31);
        expect_read(8'h04, 32'h0001_0006);
        expect_read(8'h28, 32'h0002_0002);
        expect_read(8'h2C, 32'h0002_0002);
        expect_read(8'h30, 32'h0810_1008);
        expect_read(8'h34, 32'h0000_0208);
        expect_read(8'h54, 32'h0800_0000);
        expect_read(8'h58, 32'h0501_3201);
        expect_read(8'h5C, -32'sd65);
        expect_read(8'h60, 32'd8);
        expect_read(8'h64, 32'd30);
        expect_read(8'h68, 32'h0212_120C);
        expect_read(8'h6C, 32'h6418_1418);
        expect_read(8'h70, 32'h0080_FFE0);
        expect_read(8'h74, 32'd20);

        // Decode errors are reported and made visible through ERROR/IRQ.
        axi_write(8'h94, 32'h0, 4'hF, AXI_DECERR, 0, 1);
        if (!irq)
            $fatal(1, "decode error did not assert irq");
        expect_read(8'h50, 32'h0000_0001);
        axi_write(8'h08, 32'h2, 4'h1, AXI_OKAY, 1, 0);
        if (irq)
            $fatal(1, "clear did not deassert irq");

        // Out-of-range cursor must fail without modifying upload counters/CRC.
        axi_write(8'h14, 32'h0003_0000, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h18, 32'h55, 4'h1, AXI_SLVERR, 0, 1);
        expect_read(8'h50, 32'h0000_0003);
        expect_read(8'h48, 32'h0);
        axi_write(8'h08, 32'h2, 4'h1, AXI_OKAY, 0, 0);

        // Layer 0, pre-major: [[60,0],[60,0]]. AW/W order is varied.
        axi_write(8'h14, 32'h0000_0000, 4'hF, AXI_OKAY, 2, 0);
        axi_write(8'h18, 32'd60, 4'h1, AXI_OKAY, 0, 2);
        axi_write(8'h18, 32'd0, 4'h1, AXI_OKAY, 1, 0);
        axi_write(8'h18, 32'd60, 4'h1, AXI_OKAY, 0, 1);
        axi_write(8'h18, 32'd0, 4'h1, AXI_OKAY, 0, 0);

        // Layer 1, pre-major: [[0,120],[0,0]].
        axi_write(8'h14, 32'h8000_0000, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h18, 32'd0, 4'h1, AXI_OKAY, 0, 0);
        axi_write(8'h18, 32'd120, 4'h1, AXI_OKAY, 0, 0);
        axi_write(8'h18, 32'd0, 4'h1, AXI_OKAY, 0, 0);
        axi_write(8'h18, 32'd0, 4'h1, AXI_OKAY, 0, 0);
        expect_read(8'h48, 32'd8);
        expect_read(8'h40, 32'h9E1A_6765);

        // Two t=0 input events. EVENT_NEURON commits and auto-increments.
        axi_write(8'h1C, 32'h0, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h20, 32'h0, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h24, 32'h0, 4'h3, AXI_OKAY, 0, 0);
        axi_write(8'h20, 32'h0, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h24, 32'h1, 4'h3, AXI_OKAY, 0, 0);
        expect_read(8'h4C, 32'd2);
        expect_read(8'h44, 32'h62CE_F72E);

        // Runtime stream length and horizon are independent of BRAM capacity.
        axi_write(8'h78, 32'd2, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h7C, 32'd1, 4'hF, AXI_OKAY, 0, 0);
        expect_read(8'h78, 32'd2);
        expect_read(8'h7C, 32'd1);

        axi_write(8'h08, 32'h1, 4'h1, AXI_OKAY, 0, 0);

        // A write during inference is rejected and recorded.
        axi_write(8'h18, 32'h0, 4'h1, AXI_SLVERR, 0, 0);
        wait (dut.result_valid_sticky);
        if (!irq)
            $fatal(1, "result/error did not assert irq");
        expect_read(8'h10, 32'd1);
        expect_read(8'h50, 32'h0000_0002);
        expect_read(8'h0C, 32'h0000_007D);
        axi_read(8'h80, AXI_OKAY, first_cycles);
        if (first_cycles == 0)
            $fatal(1, "inference cycle telemetry is zero");
        expect_read(8'h84, 32'd0);
        expect_read(8'h88, 32'd2);
        expect_read(8'h8C, 32'd1);
        expect_read(8'h90, 32'd1);
        expect_read(8'h94, 32'h0002_0002);
        axi_read_phase(8'h98, phase_low_word);
        axi_read_phase(8'h9C, phase_high_word);
        synapse_cycles = {phase_high_word, phase_low_word};
        axi_read_phase(8'hA0, phase_low_word);
        axi_read_phase(8'hA4, phase_high_word);
        hidden_update_cycles = {phase_high_word, phase_low_word};
        axi_read_phase(8'hA8, phase_low_word);
        axi_read_phase(8'hAC, phase_high_word);
        hidden_routing_cycles = {phase_high_word, phase_low_word};
        axi_read_phase(8'hB0, phase_low_word);
        axi_read_phase(8'hB4, phase_high_word);
        output_update_cycles = {phase_high_word, phase_low_word};
        axi_read_phase(8'hB8, phase_low_word);
        axi_read_phase(8'hBC, phase_high_word);
        control_overhead_cycles = {phase_high_word, phase_low_word};
        if (synapse_cycles + hidden_update_cycles + hidden_routing_cycles +
            output_update_cycles + control_overhead_cycles !== first_cycles)
            $fatal(1, "phase cycles do not sum to inference cycles");
        if (synapse_cycles == 0 || hidden_update_cycles == 0 ||
            hidden_routing_cycles == 0 || output_update_cycles == 0 ||
            control_overhead_cycles == 0)
            $fatal(1, "one or more phase counters are zero");

        // Clear acknowledges result/error and resets upload verification state.
        axi_write(8'h08, 32'h2, 4'h1, AXI_OKAY, 0, 0);
        expect_read(8'h0C, 32'h0000_0061);
        expect_read(8'h40, 32'h0);
        expect_read(8'h44, 32'h0);
        expect_read(8'h48, 32'h0);
        expect_read(8'h4C, 32'h0);
        axi_read(8'hC0, AXI_DECERR, ignored);

        // An empty rate-coded stream must still execute the configured horizon.
        axi_write(8'h78, 32'd0, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h7C, 32'd2, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h08, 32'h1, 4'h1, AXI_OKAY, 0, 0);
        wait (dut.done_sticky);
        expect_read(8'h0C, 32'h0000_0065);
        axi_read(8'h80, AXI_OKAY, first_cycles);
        if (first_cycles == 0)
            $fatal(1, "empty inference cycle telemetry is zero");
        expect_read(8'h88, 32'd0);
        expect_read(8'h8C, 32'd0);
        expect_read(8'h90, 32'd0);

        // A non-empty stream may also use less than the physical capacity.
        axi_write(8'h08, 32'h2, 4'h1, AXI_OKAY, 0, 0);
        axi_write(8'h1C, 32'h0, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h20, 32'h0, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h24, 32'h0, 4'h3, AXI_OKAY, 0, 0);
        axi_write(8'h78, 32'd1, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h7C, 32'd2, 4'hF, AXI_OKAY, 0, 0);
        axi_write(8'h08, 32'h1, 4'h1, AXI_OKAY, 0, 0);
        wait (dut.done_sticky);
        expect_read(8'h0C, 32'h0000_0065);
        axi_read(8'h80, AXI_OKAY, first_cycles);
        if (first_cycles == 0)
            $fatal(1, "non-empty inference cycle telemetry is zero");
        expect_read(8'h88, 32'd1);
        expect_read(8'h8C, 32'd0);
        expect_read(8'h90, 32'd0);

        // START acknowledges stale DONE and replays the retained event stream
        // without resetting its upload count/CRC.
        axi_write(8'h08, 32'h1, 4'h1, AXI_OKAY, 0, 0);
        if (dut.done_sticky)
            $fatal(1, "replay START did not clear stale DONE");
        expect_read(8'h4C, 32'd1);
        wait (dut.done_sticky);
        axi_read(8'h80, AXI_OKAY, replay_cycles);
        if (replay_cycles !== first_cycles)
            $fatal(1, "replay cycles %0d differ from first run %0d",
                   replay_cycles, first_cycles);
        expect_read(8'h88, 32'd1);
        expect_read(8'h8C, 32'd0);
        expect_read(8'h90, 32'd0);

        $display("RESULT: PASS");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout");
    end
endmodule
