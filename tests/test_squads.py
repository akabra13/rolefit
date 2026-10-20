from rolefit.squads import norm, parse_squad, season_links, split_params


PAGE = """
== History ==
Some prose with [[Somebody]].
== Players ==
=== Current squad ===
{{Fs start}}
{{Fs player|no=1|nat=ESP|pos=GK|name=[[David Raya]]}}
{{Fs player|no=4|nat=ENG|pos=DF|name=[[Ben White (footballer)|Ben White]]}}
{{Fs mid}}
{| class="wikitable"
{{Football squad player sort |no=2 |nat=MEX |pos=DF |first=Reyna |last=Reyes}}
|}
{{Fs end}}
=== Out on loan ===
{{Fs player|no=30|nat=ENG|pos=FW|name=[[Loanee Player]]}}
== Staff ==
"""


def test_parse_squad_reads_nested_templates_and_stops_before_loans():
    sq = parse_squad(PAGE)
    names = [p["name"] for p in sq]
    assert names == ["David Raya", "Ben White", "Reyna Reyes"]
    assert sq[1]["title"] == "Ben White (footballer)" and sq[2]["title"] is None


def test_split_params_respects_links():
    assert split_params("Fs player|name=[[A (b)|A]]|pos=DF") == ["Fs player", "name=[[A (b)|A]]", "pos=DF"]


def test_season_links_skip_prose():
    text = """== Teams ==
Promoted [[Prose Club]] replaced [[Relegated Club]].
=== Stadium and location ===
{| class="wikitable"
|-
| [[Arsenal F.C.|Arsenal]] || London || [[Emirates Stadium]]
|}
{{location map~ |England |label=[[Chelsea F.C.|Chelsea]]}}
"""
    links = season_links(text)
    assert "Arsenal F.C." in links and "Chelsea F.C." in links and "Prose Club" not in links


def test_norm():
    assert norm("Rodrigo Hernández Cascante") == "rodrigo hernandez cascante"
    assert norm("Ben White (footballer)") == "ben white"
