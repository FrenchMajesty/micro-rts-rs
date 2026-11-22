import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import ai.reward.AttackRewardFunction;
import ai.reward.ProduceBuildingRewardFunction;
import ai.reward.ProduceCombatUnitRewardFunction;
import ai.reward.ProduceWorkerRewardFunction;
import ai.reward.ResourceGatherRewardFunction;
import ai.reward.RewardFunctionInterface;
import ai.reward.WinLossRewardFunction;
import rts.units.UnitTypeTable;
import tests.JNIGridnetClientSelfPlay;

/** Keeps the upstream JVM outside the Python process. */
public final class GridNetBridge {
    private static int[][] readOrders(DataInputStream input) throws Exception {
        int[][] orders = new int[input.readInt()][8];
        for (int[] order : orders) {
            for (int index = 0; index < order.length; index++) {
                order[index] = input.readInt();
            }
        }
        return orders;
    }

    private static void writeState(DataOutputStream output, JNIGridnetClientSelfPlay client)
            throws Exception {
        boolean terminal = client.gs.gameover();
        output.writeInt(client.gs.getTime());
        output.writeBoolean(terminal);
        output.writeByte(terminal ? client.gs.winner() : -1);
        for (int player = 0; player < 2; player++) {
            for (int[][] plane : client.getResponse(player).observation) {
                for (int[] row : plane) {
                    for (int value : row) {
                        output.writeInt(value);
                    }
                }
            }
            for (int[][] row : client.getMasks(player)) {
                for (int[] cell : row) {
                    for (int value : cell) {
                        output.writeByte(value);
                    }
                }
            }
        }
        output.flush();
    }

    public static void main(String[] args) throws Exception {
        DataInputStream input = new DataInputStream(new BufferedInputStream(System.in));
        DataOutputStream output = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(FileDescriptor.out)));
        System.setOut(System.err);
        JNIGridnetClientSelfPlay client = new JNIGridnetClientSelfPlay(
                new RewardFunctionInterface[] {
                    new WinLossRewardFunction(), new ResourceGatherRewardFunction(),
                    new ProduceWorkerRewardFunction(), new ProduceBuildingRewardFunction(),
                    new AttackRewardFunction(), new ProduceCombatUnitRewardFunction()
                },
                "", args[0], new UnitTypeTable(), false);
        try {
            int command;
            while ((command = input.read()) != -1) {
                if (command == 'r') {
                    client.reset();
                } else if (command == 's') {
                    client.gameStep(readOrders(input), readOrders(input));
                } else {
                    throw new IllegalArgumentException("Unknown GridNet bridge command " + command);
                }
                writeState(output, client);
            }
        } finally {
            client.close();
        }
    }
}
