# How to run

You'll need a json file that encodes past matches (see data/matches.json) like this:

```json
  {
    "teams": [
      [
        "levi",
        "andre",
        "max",
        "szimi"
      ],
      [
        "dom",
        "mayo",
        "ala",
        "oli"
      ]
    ],
    "score": [
      13,
      7
    ]
  }
```

Then run 

```
python3 footyeast.py --players player1 player2 player 3 --data_path your_json_file.json
```

(You might need to `python3 setup.py install` to get the dependencies)