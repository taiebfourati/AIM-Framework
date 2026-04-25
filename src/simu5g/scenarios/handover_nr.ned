// ==========================================================================
// handover_nr.ned — Simu5G NR Handover Scenario for RTP Observer Framework
//
// This NED file defines a multi-cell 5G NR standalone network with UE
// mobility for testing the AI Management Framework's drift/poisoning
// detection under realistic radio conditions.
//
// Network topology:
//   - 3 gNBs in triangular layout (ISD = 500m, 3GPP Urban Macro)
//   - 10 UEs with linear/random-waypoint mobility
//   - Uplink + Downlink VoIP & CBR traffic
//   - X2/Xn handover enabled between gNBs
//
// KPI signals recorded:
//   - SS-RSRP, SS-SINR per UE per TTI
//   - DL/UL throughput (RLC-level)
//   - End-to-end latency (application-level)
//   - Handover events (source gNB, target gNB, cause)
//
// Reference: Simu5G v1.4.x, OMNeT++ 6.x, INET 4.5+
// ==========================================================================

package simu5g.simulations.nr.handover_rtp;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;

import simu5g.common.binder.Binder;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.nodes.NR.gNodeB;
import simu5g.nodes.NR.NRUe;
import simu5g.nodes.backgroundCell.BackgroundCell;
import simu5g.world.radio.LteChannelControl;

network HandoverNR
{
    parameters:
        int numUe = default(10);
        int numBgCells = default(6);
        @display("bgb=1200,800");

    submodules:
        channelControl: LteChannelControl {
            @display("p=50,25;is=s");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=180,25;is=s");
        }
        binder: Binder {
            @display("p=310,25;is=s");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=440,25;is=s");
        }
        router: Router {
            @display("p=600,150;is=l");
        }
        server: StandardHost {
            @display("p=600,50;is=l");
        }

        // --- 3 gNBs in triangular layout (ISD ≈ 500m) ---
        gnb1: gNodeB {
            @display("p=300,400;is=vl");
        }
        gnb2: gNodeB {
            @display("p=700,400;is=vl");
        }
        gnb3: gNodeB {
            @display("p=500,150;is=vl");
        }

        // --- Mobile UEs ---
        ue[numUe]: NRUe {
            @display("p=400,500;is=l");
        }

        // --- Background cells for inter-cell interference ---
        bgCell[numBgCells]: BackgroundCell {
            @display("p=100,700;is=s");
        }

    connections allowunconnected:
        server.pppg++ <--> { datarate = 10Gbps; delay = 1ms; } <--> router.pppg++;
        router.pppg++ <--> { datarate = 10Gbps; delay = 1ms; } <--> gnb1.ppp;
        router.pppg++ <--> { datarate = 10Gbps; delay = 1ms; } <--> gnb2.ppp;
        router.pppg++ <--> { datarate = 10Gbps; delay = 1ms; } <--> gnb3.ppp;

        // X2/Xn connections for handover
        gnb1.x2++ <--> { datarate = 10Gbps; delay = 0.1ms; } <--> gnb2.x2++;
        gnb1.x2++ <--> { datarate = 10Gbps; delay = 0.1ms; } <--> gnb3.x2++;
        gnb2.x2++ <--> { datarate = 10Gbps; delay = 0.1ms; } <--> gnb3.x2++;
}
