"""Team-name mappings, so the same club has one name across seasons and sources."""

# Same club under two names inside Matches.csv. Only true duplicates belong here:
# Las Palmas / U.Las Palmas, Extremadura / Extremadura UD and Granada / Granada 74 look similar
# but are different clubs, so they are deliberately not mapped.
MATCHES_NAME_FIXES: dict[str, str] = {"Leonesa": "Cultural Leonesa"}
