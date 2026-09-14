"""Inject data/cloud.json and data/activity.json into viewer_template.html -> viewer.html."""
import os
t = open("viewer_template.html").read()
t = t.replace("/*__DATA__*/null", open("data/cloud.json").read())
act = open("data/activity.json").read() if os.path.exists("data/activity.json") else "null"
t = t.replace("/*__ACTIVITY__*/null", act)
open("viewer.html", "w").write(t)
print("viewer.html", round(len(t) / 1e6, 2), "MB")
