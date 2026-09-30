`timescale 1ns/1ps

module tb_izhikevich_array;
    localparam int NUM_NEURONS = 3;
    localparam int MEM_WIDTH = 24;
    localparam int CURRENT_WIDTH = 24;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic clear = 1'b0;
    logic update_valid = 1'b0;
    logic direct_start = 1'b0;
    logic signed [CURRENT_WIDTH-1:0] syn_current [NUM_NEURONS];
    logic [NUM_NEURONS-1:0] serial_spike;
    logic signed [MEM_WIDTH-1:0] serial_membrane [NUM_NEURONS];
    logic serial_done;
    logic [NUM_NEURONS-1:0] direct_done;
    logic [NUM_NEURONS-1:0] direct_spike;
    logic [NUM_NEURONS-1:0] direct_spike_seen;
    logic signed [MEM_WIDTH-1:0] direct_membrane [NUM_NEURONS];
    logic signed [MEM_WIDTH-1:0] direct_recovery [NUM_NEURONS];
    integer i;

    always #5 clk = ~clk;

    neuron_array #(
        .NUM_NEURONS(NUM_NEURONS),
        .MEM_WIDTH(MEM_WIDTH),
        .CURRENT_WIDTH(CURRENT_WIDTH),
        .NEURON_MODEL(1)
    ) dut (
        .clk(clk),
        .rst(rst),
        .clear(clear),
        .update_valid(update_valid),
        .syn_current(syn_current),
        .spike(serial_spike),
        .membrane(serial_membrane),
        .done(serial_done)
    );

    for (genvar g = 0; g < NUM_NEURONS; g++) begin : gen_direct
        izhikevich_neuron #(
            .MEM_WIDTH(MEM_WIDTH),
            .CURRENT_WIDTH(CURRENT_WIDTH)
        ) u_direct (
            .clk(clk),
            .rst(rst),
            .clear(clear),
            .state_load(1'b0),
            .state_voltage('0),
            .state_recovery('0),
            .start(direct_start),
            .syn_current(syn_current[g]),
            .done(direct_done[g]),
            .spike(direct_spike[g]),
            .membrane(direct_membrane[g]),
            .recovery(direct_recovery[g])
        );
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst || clear)
            direct_spike_seen <= '0;
        else if (update_valid)
            direct_spike_seen <= '0;
        else
            direct_spike_seen <= direct_spike_seen | direct_spike;
    end

    task automatic run_update;
        begin
            @(negedge clk);
            update_valid = 1'b1;
            direct_start = 1'b1;
            @(negedge clk);
            update_valid = 1'b0;
            direct_start = 1'b0;
            wait (serial_done);
            #1;
            for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                if ($signed(serial_membrane[i]) !==
                        $signed(direct_membrane[i]))
                    $fatal(1, "neuron %0d membrane=%0d expected=%0d",
                        i, serial_membrane[i], direct_membrane[i]);
                if ($signed(dut.gen_izh_serial.recovery_state[i]) !==
                        $signed(direct_recovery[i]))
                    $fatal(1, "neuron %0d recovery=%0d expected=%0d",
                        i, dut.gen_izh_serial.recovery_state[i],
                        direct_recovery[i]);
                if (serial_spike[i] !== direct_spike_seen[i])
                    $fatal(1, "neuron %0d spike=%0b expected=%0b",
                        i, serial_spike[i], direct_spike_seen[i]);
            end
            @(negedge clk);
        end
    endtask

    initial begin
        syn_current[0] = 24'sd10;
        syn_current[1] = 24'sd0;
        syn_current[2] = -24'sd5;

        repeat (3) @(posedge clk);
        rst = 1'b0;
        #1;
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin
            if ($signed(serial_membrane[i]) !== -16640)
                $fatal(1, "neuron %0d initial membrane=%0d", i,
                    serial_membrane[i]);
            if ($signed(dut.gen_izh_serial.recovery_state[i]) !== -3328)
                $fatal(1, "neuron %0d initial recovery=%0d", i,
                    dut.gen_izh_serial.recovery_state[i]);
        end

        run_update();
        run_update();
        run_update();
        run_update();

        $display("RESULT: PASS");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout");
    end
endmodule
