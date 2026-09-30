`timescale 1ns/1ps

module tb_neuron_array;


    //----------------------------------------------------------------------
    // Parameters
    //----------------------------------------------------------------------
    
    localparam NUM_NEURONS   = 4;
    localparam MEM_WIDTH     = 16;
    localparam CURRENT_WIDTH = 16;

    localparam THRESHOLD  = 100;
    localparam RESET_POT  = 0;
    localparam LEAK_SHIFT = 2;


    //----------------------------------------------------------------------
    // Clock / reset
    //----------------------------------------------------------------------
    
    logic clk;
    logic rst;
    logic clear;


    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("./sim/sim_neuron_array.vcd");
        $dumpvars(0, tb_neuron_array);
    end


    //----------------------------------------------------------------------
    // DUT signals
    //----------------------------------------------------------------------
    
    logic update_valid;

    logic signed [CURRENT_WIDTH-1:0]
        syn_current [NUM_NEURONS];


    logic [NUM_NEURONS-1:0]
        spike;


    logic signed [MEM_WIDTH-1:0]
        membrane [NUM_NEURONS];


    logic done;
    logic spike0_seen;

    always @(posedge clk) begin
        if (rst)
            spike0_seen <= 1'b0;
        else if (spike[0])
            spike0_seen <= 1'b1;
    end


    //----------------------------------------------------------------------
    // DUT
    //----------------------------------------------------------------------
    
    neuron_array #(
        .NUM_NEURONS(NUM_NEURONS),
        .MEM_WIDTH(MEM_WIDTH),
        .CURRENT_WIDTH(CURRENT_WIDTH),
        .THRESHOLD(THRESHOLD),
        .RESET_POT(RESET_POT),
        .LEAK_SHIFT(LEAK_SHIFT)
    )
    dut
    (
        .clk(clk),
        .rst(rst),
        .clear(clear),

        .update_valid(update_valid),

        .syn_current(syn_current),

        .spike(spike),
        .membrane(membrane),

        .done(done)
    );


    //----------------------------------------------------------------------
    // Helper task
    //----------------------------------------------------------------------
    
    task start_update;
        begin

            @(negedge clk);
            update_valid = 1'b1;

            @(negedge clk);
            update_valid = 1'b0;

        end
    endtask



    //----------------------------------------------------------------------
    // Test sequence
    //----------------------------------------------------------------------
    
    initial begin

        integer i;


        //--------------------------------------------------------------
        // Initial values
        //--------------------------------------------------------------

        rst = 1;
        clear = 0;
        update_valid = 0;

        for(i=0;i<NUM_NEURONS;i=i+1)
            syn_current[i] = 0;


        //--------------------------------------------------------------
        // Reset
        //--------------------------------------------------------------

        repeat(3)
            @(posedge clk);

        rst = 0;


        //--------------------------------------------------------------
        // Test 1:
        // Small currents -> no spike
        //--------------------------------------------------------------

        $display("\nTEST 1: integration without spike");


        syn_current[0] = 10;
        syn_current[1] = 20;
        syn_current[2] = 30;
        syn_current[3] = 40;


        start_update();


        wait(done);


        @(posedge clk);


        for(i=0;i<NUM_NEURONS;i=i+1)
        begin
            $display(
                "Neuron %0d membrane=%0d spike=%b",
                i,
                membrane[i],
                spike[i]
            );
        end



        //--------------------------------------------------------------
        // Test 2:
        // Threshold crossing
        //--------------------------------------------------------------

        $display("\nTEST 2: spike generation");


        syn_current[0] = 150;
        syn_current[1] = 0;
        syn_current[2] = 0;
        syn_current[3] = 0;

        spike0_seen = 1'b0;


        start_update();


        wait(done);


        @(posedge clk);


        if(spike0_seen)
            $display("PASS: neuron 0 generated spike");
        else
            $fatal(1, "neuron 0 did not spike");


        //--------------------------------------------------------------
        // Test 3:
        // Check reset after spike
        //--------------------------------------------------------------

        if(membrane[0] == RESET_POT)
            $display("PASS: membrane reset");
        else
            $fatal(1, "membrane=%0d, expected reset", membrane[0]);



        //--------------------------------------------------------------
        // Test 4:
        // Check repeated updates
        //--------------------------------------------------------------

        $display("\nTEST 4: repeated integration");


        syn_current[0] = 60;


        repeat(5)
        begin

            start_update();

            wait(done);

            @(posedge clk);

            $display(
                "membrane=%0d",
                membrane[0]
            );

        end



        //--------------------------------------------------------------
        // Finish
        //--------------------------------------------------------------

        $display("\nSimulation finished");

        #20;

        $finish;

    end



endmodule
