from src.utils.parse_whiteboard_xml import parse_whiteboard_xml

traj = parse_whiteboard_xml(
    "data/IAM_OnDB/trajectories/b04-334z-01.xml"
)

print(traj.shape)
print(traj[:5])
