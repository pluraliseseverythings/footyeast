from copy import copy


def all_teams_of_size(players_set, size):
    teams = set()
    if size == 0:
        teams.add(frozenset())
    else:
        for p in players_set:
            removed_players_set = copy(players_set)
            removed_players_set.remove(p)
            smaller_teams = all_teams_of_size(removed_players_set, size - 1)
            for s in smaller_teams:
                unfrozen_s = set(copy(s))
                unfrozen_s.add(p)
                teams.add(frozenset(unfrozen_s))
    return teams


def all_possible_matches(players):
    players_set = set(players)
    matches = set()
    all_teams = all_teams_of_size(players_set, len(players_set) / 2)
    for t in all_teams:
        other_t = frozenset(players_set - t)
        if hash(t) > hash(other_t):
            matches.add(frozenset([t, other_t]))
    return matches
