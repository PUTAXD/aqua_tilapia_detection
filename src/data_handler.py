import pandas as pd
import matplotlib.pyplot as plt
import os


def save_fish_positions_to_csv(fish_positions_log, output_csv):
    if fish_positions_log:
        df = pd.DataFrame(fish_positions_log, columns=[
                          'timestamp', 'fish_id', 'x_coordinate', 'y_coordinate'])
        df.to_csv(output_csv, index=False)
        print(f"📝 Fish positions saved to {output_csv}")


def plot_average_speed(speed_log, output_graph_speed):
    if speed_log:
        times = [t - speed_log[0][0] for t, _ in speed_log]
        speeds = [s for _, s in speed_log]
        plt.figure(figsize=(10, 5))
        plt.plot(times, speeds, marker="o", color="blue", linewidth=2)
        plt.title("Average Speed Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Average Speed (px/s)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_graph_speed)
        print(f"📊 Average speed graph saved to {output_graph_speed}")


def plot_fish_trajectories(output_csv, output_graph_trajectory):
    if os.path.exists(output_csv):
        df_positions = pd.read_csv(output_csv)
        plt.figure(figsize=(12, 8))
        for fish_id in df_positions['fish_id'].unique():
            fish_data = df_positions[df_positions['fish_id'] == fish_id]
            plt.plot(fish_data['x_coordinate'], fish_data['y_coordinate'],
                     marker='.', label=f'Fish ID {fish_id}')
        plt.title("Fish Trajectories")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_graph_trajectory)
        print(f"📊 Fish trajectories graph saved to {output_graph_trajectory}")
