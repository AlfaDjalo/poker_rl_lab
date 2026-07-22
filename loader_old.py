import yaml
from pathlib import Path

from rules.game_definition import GameDefinition
from rules.game_rules import GameRules, PointDefinition

from poker_eval import ScoreType, ShowdownType


BASE_PATH = Path(__file__).parent
DEFAULT_STREET_NAMES = ["pre-flop", "flop", "turn", "river"]

def _default_street_names(index: int) -> str:
    if index < len(DEFAULT_STREET_NAMES):
        return DEFAULT_STREET_NAMES[index]
    return f"street_{index}"

def load_game(name):

    path = BASE_PATH / f"{name}.yaml"

    with open(path) as f:
        data = yaml.safe_load(f)

    # -----------------------------
    # board layout
    # -----------------------------

    raw_streets = data["board_layout"]["streets"]

    # Keys are street indices (highest = last/showdown street).
    # Streets with no nodes are implicitly zero cards dealt.   
    max_index = max(int(k) for k in raw_streets)

    street_nodes = []
    street_names_raw = data["board_layout"].get("street_names", {})

    street_nodes_list = []
    street_names = []

    for i in range(max_index + 1):
        nodes = raw_streets.get(i, raw_streets.get(str(i), []))
        street_nodes_list.append(list(nodes))
        street_name = street_names_raw.get(i, raw_streets.get(str(i), _default_street_names(i)))
        street_names.append(street_name)

    street_nodes = street_nodes_list
    board_cards_per_street = [len(s) for s in street_nodes]
        
    board_cards_per_street = [len(s) for s in street_nodes]
    
    node_count = data["board_layout"]["nodes"]
    # -----------------------------
    # points
    # ----------------------------- 

    points = []

    score_types = set()

    for p in data["points"]:

        score_type = getattr(ScoreType, p["score_type"])

        score_types.add(score_type)

        override_str = p.get("showdown_type_override")
        showdown_type_override = (
            getattr(ShowdownType, override_str) if override_str else None
        )

        points.append(

            PointDefinition(
                name=p["name"],
                score_type=score_type,
                node_sets=[tuple(ns) for ns in p["node_sets"]],
                showdown_type_override=showdown_type_override,
                scoop_from=p.get("scoop_from")
            )
        )   

    score_types = list(score_types)

    # -----------------------------
    # showdown
    # -----------------------------

    showdown_type = getattr(ShowdownType, data["showdown"]["type"])

    # -----------------------------
    # betting
    # -----------------------------

    betting = data["betting"]

    betting_type = betting.get("type", "holdem")
    small_blind = betting.get("small_blind", 0)
    big_blind = betting.get("big_blind", 0)
    ante = betting.get("ante", 0)

    # -----------------------------
    # GameDefinition
    # -----------------------------
    #     
    game_def = GameDefinition(
        hole_cards=data["hole_cards"],
        board_cards_per_street=board_cards_per_street,
        street_nodes=street_nodes,
        street_names=street_names,
        score_types=score_types,
        node_count=node_count,
        # low_qualifier=data.get("low_qualifier"),
        betting_type=betting_type,
        small_blind=small_blind,
        big_blind=big_blind,
        ante=ante,
        layout_name=data.get("layout_name"),
        game_name=name,
    )

    # -----------------------------
    # GameRules
    # -----------------------------

    rules = GameRules(
        score_types=score_types,
        showdown_type=showdown_type,
        points=points,
        payout_type=data.get("payout_type", "points"),
        low_qualifier=data.get("low_qualifier", 8),
        no_qualify_action=data.get("no_qualify_action", "scoop"),
    )

    return game_def, rules