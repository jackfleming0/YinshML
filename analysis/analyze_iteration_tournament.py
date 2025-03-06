#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_tournament(model_name):
    """
    1. Constructs the path to the tournament_history.json file for the given model.
    2. Reads and aggregates matchup data to compute overall win rates and average game length.
    3. Produces two summary tables:
         - A directional matchup table (printed to terminal).
         - A per-iteration overall performance table.
    4. Additionally, creates a detailed matchup table where rows are iterations and columns (per opponent)
       are split into three metrics: win rate when playing as white, win rate when playing as black,
       and average game length.
    5. The detailed table is styled with a gentle heat map and saved as HTML.
    """
    # Construct the path to the JSON file.
    json_path = os.path.expanduser(f"~/PycharmProjects/YinshML/checkpoints/combined/{model_name}/tournament_history.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Aggregate directional matchup statistics.
    matchup_stats = defaultdict(lambda: {
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "total_game_length": 0.0,
        "games_count": 0
    })

    for tournament_key, tournament_info in data.items():
        results = tournament_info["results"]
        for match in results:
            white = match["white_model"]
            black = match["black_model"]
            w_wins = match["white_wins"]
            b_wins = match["black_wins"]
            draws = match["draws"]
            length = match["avg_game_length"]
            num_games = w_wins + b_wins + draws
            key = (white, black)
            matchup_stats[key]["white_wins"] += w_wins
            matchup_stats[key]["black_wins"] += b_wins
            matchup_stats[key]["draws"] += draws
            matchup_stats[key]["total_game_length"] += length * num_games
            matchup_stats[key]["games_count"] += num_games

    # Build a DataFrame from the aggregated dictionary.
    rows = []
    for (white_model, black_model), stats in matchup_stats.items():
        if stats["games_count"] == 0:
            continue
        total = stats["games_count"]
        white_win_rate = stats["white_wins"] / total
        black_win_rate = stats["black_wins"] / total
        draw_rate = stats["draws"] / total
        avg_length = stats["total_game_length"] / total
        rows.append({
            "white_model": white_model,
            "black_model": black_model,
            "white_wins": stats["white_wins"],
            "black_wins": stats["black_wins"],
            "draws": stats["draws"],
            "total_games": total,
            "white_win_rate": round(white_win_rate, 3),
            "black_win_rate": round(black_win_rate, 3),
            "draw_rate": round(draw_rate, 3),
            "avg_game_length": round(avg_length, 2),
        })
    df = pd.DataFrame(rows)
    df.sort_values(by=["white_model", "black_model"], inplace=True, ignore_index=True)

    print("Matchup Results:")
    print(df.to_string(index=False))

    # Create per-iteration overall performance summaries.
    # Note: Rename total_games columns to avoid duplicates.
    df_white_summary = (
        df.groupby("white_model")
          .agg({"white_wins": "sum", "total_games": "sum"})
          .rename(columns={"white_wins": "total_white_wins", "total_games": "total_games_white"})
    )
    df_white_summary["white_win_rate_overall"] = df_white_summary["total_white_wins"] / df_white_summary["total_games_white"]

    df_black_summary = (
        df.groupby("black_model")
          .agg({"black_wins": "sum", "total_games": "sum"})
          .rename(columns={"black_wins": "total_black_wins", "total_games": "total_games_black"})
    )
    df_black_summary["black_win_rate_overall"] = df_black_summary["total_black_wins"] / df_black_summary["total_games_black"]

    combined = pd.concat([df_white_summary, df_black_summary], axis=1).fillna(0)
    combined["total_wins"] = combined["total_white_wins"] + combined["total_black_wins"]
    combined["total_games"] = combined["total_games_white"] + combined["total_games_black"]
    combined["overall_win_rate"] = combined["total_wins"] / combined["total_games"]

    print("\nPer-Iteration Overall Performance:")
    print(combined.to_string())

    # Save these summaries to CSV.
    output_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(output_dir, "matchup_stats.csv"), index=False)
    combined.to_csv(os.path.join(output_dir, "iteration_performance.csv"))

    # Build and save the detailed matchup table with multi-level columns.
    detailed_styled = create_detailed_matchup_table(data)
    html_path = os.path.join(output_dir, "detailed_matchup_table.html")
    with open(html_path, "w") as f:
        f.write(detailed_styled.to_html())
    print(f"\nDetailed matchup table saved as HTML to: {html_path}")


def create_detailed_matchup_table(tournament_data):
    """
    Creates a table where rows are iterations and for each opponent (column) there are three sub-columns:
      - "as white": win rate when the row iteration played as white against that opponent.
      - "as black": win rate when the row iteration played as black against that opponent.
      - "avg game length": weighted average game length across both directions.
    In this table, only the win rate columns are heatmapped.

    Note: For example, a cell in row "iteration_0" under column ("iteration_3", "as white")
          gives the win rate when iteration_0 (the row) played as white against iteration_3.
    """
    # First, re-aggregate matchup data as in analyze_tournament.
    matchup_stats = {}
    for _, info in tournament_data.items():
        for match in info["results"]:
            key = (match["white_model"], match["black_model"])
            if key not in matchup_stats:
                matchup_stats[key] = {"white_wins": 0, "black_wins": 0, "draws": 0, "total_game_length": 0.0,
                                      "total_games": 0}
            num_games = match["white_wins"] + match["black_wins"] + match["draws"]
            matchup_stats[key]["white_wins"] += match["white_wins"]
            matchup_stats[key]["black_wins"] += match["black_wins"]
            matchup_stats[key]["draws"] += match["draws"]
            matchup_stats[key]["total_game_length"] += match["avg_game_length"] * num_games
            matchup_stats[key]["total_games"] += num_games

    # Compute derived metrics for each directional matchup.
    derived = {}
    iterations_set = set()
    for (i, j), stats in matchup_stats.items():
        iterations_set.add(i)
        iterations_set.add(j)
        if stats["total_games"] > 0:
            derived[(i, j)] = {
                "white_win_rate": stats["white_wins"] / stats["total_games"],
                "black_win_rate": stats["black_wins"] / stats["total_games"],
                "avg_game_length": stats["total_game_length"] / stats["total_games"],
                "total_games": stats["total_games"]
            }
        else:
            derived[(i, j)] = {
                "white_win_rate": np.nan,
                "black_win_rate": np.nan,
                "avg_game_length": np.nan,
                "total_games": 0
            }
    iterations = sorted(iterations_set, key=lambda s: int(s.split('_')[1]))

    # Build a nested dictionary: table[row][opponent] = dict of metrics.
    table = {i: {} for i in iterations}
    for i in iterations:
        for j in iterations:
            if i == j:
                table[i][j] = {"as white": np.nan, "as black": np.nan, "avg game length": np.nan}
            else:
                rec_white = derived.get((i, j))
                rec_black = derived.get((j, i))
                val_white = rec_white["white_win_rate"] if rec_white is not None else np.nan
                val_black = rec_black["black_win_rate"] if rec_black is not None else np.nan
                if rec_white is not None and rec_black is not None:
                    total_games = rec_white["total_games"] + rec_black["total_games"]
                    avg_length = ((rec_white["avg_game_length"] * rec_white["total_games"]) +
                                  (rec_black["avg_game_length"] * rec_black["total_games"])) / total_games
                elif rec_white is not None:
                    avg_length = rec_white["avg_game_length"]
                elif rec_black is not None:
                    avg_length = rec_black["avg_game_length"]
                else:
                    avg_length = np.nan
                table[i][j] = {"as white": val_white, "as black": val_black, "avg game length": avg_length}

    # Build a DataFrame with MultiIndex columns.
    data = {}
    for opp in iterations:
        col_as_white = []
        col_as_black = []
        col_avg = []
        for i in iterations:
            cell = table[i][opp]
            col_as_white.append(cell["as white"])
            col_as_black.append(cell["as black"])
            col_avg.append(cell["avg game length"])
        data[(opp, "as white")] = col_as_white
        data[(opp, "as black")] = col_as_black
        data[(opp, "avg game length")] = col_avg
    df_table = pd.DataFrame(data, index=iterations)
    df_table.columns = pd.MultiIndex.from_tuples(df_table.columns)

    # Style the DataFrame: apply heat mapping only to win rate columns.
    win_rate_cols = [col for col in df_table.columns if col[1] in ["as white", "as black"]]
    styled = df_table.style.format("{:.2f}")
    styled = styled.background_gradient(subset=win_rate_cols, cmap="YlGnBu", vmin=0, vmax=1)
    styled = styled.set_caption(
        "Detailed Matchup Table\nRows = Iteration; Columns per Opponent: 'as white' (row iteration playing as white), 'as black' (row iteration playing as black), and 'avg game length'")
    return styled

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_name>")
        sys.exit(1)
    model_name = sys.argv[1]
    analyze_tournament(model_name)

if __name__ == "__main__":
    main()