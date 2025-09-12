import xml.etree.ElementTree as ET

def parse_urdf(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    links = {}
    joints = []

    for link in root.findall("link"):
        link_name = link.get("name")
        links[link_name] = {"visual": None, "collision": None}

    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        xyz = origin.get("xyz") if origin is not None else "0 0 0"
        rpy = origin.get("rpy") if origin is not None else "0 0 0"
        joints.append({"name": joint_name, "parent": parent, "child": child, "xyz": xyz, "rpy": rpy})

    return links, joints

def visualize_links(joints):
    print("Robot Connection Visualization:")
    for joint in joints:
        print(f"{joint['parent']}")
        print(f"   |")
        print(f" joint {joint['name']} (origin: {joint['xyz']}, rpy: {joint['rpy']})")
        print(f"   |")
        print(f"{joint['child']}")
        print()

if __name__ == "__main__":
    urdf_file = "/home/ryan/Documents/GitHub/AirExo-2-test/airexo/urdf_models/robot/our_robot.urdf"
    links, joints = parse_urdf(urdf_file)
    visualize_links(joints)