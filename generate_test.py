import numpy as np
import math
import torch

class JointDescriptionGenerator:
    def __init__(self):
        self.joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
        
        # 关节索引映射
        self.joint_idx = {name: i for i, name in enumerate(self.joint_names)}
        
        # 定义身体部位连接关系
        self.connections = {
            'spine': ['pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head'],
            'left_arm': ['left_shoulder', 'left_elbow', 'left_wrist'],
            'right_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
            'left_leg': ['left_hip', 'left_knee', 'left_ankle', 'left_foot'],
            'right_leg': ['right_hip', 'right_knee', 'right_ankle', 'right_foot']
        }

    def calculate_angle(self, p1, p2, p3):
        """计算三点间的角度"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        return angle

    def get_limb_angle(self, joints, joint1_name, joint2_name, joint3_name):
        """获取肢体弯曲角度"""
        idx1 = self.joint_idx[joint1_name]
        idx2 = self.joint_idx[joint2_name]
        idx3 = self.joint_idx[joint3_name]
        
        return self.calculate_angle(joints[idx1], joints[idx2], joints[idx3])

    def describe_arm_position(self, joints, side):
        """描述手臂位置"""
        shoulder_idx = self.joint_idx[f'{side}_shoulder']
        elbow_idx = self.joint_idx[f'{side}_elbow']
        wrist_idx = self.joint_idx[f'{side}_wrist']
        
        shoulder = joints[shoulder_idx]
        elbow = joints[elbow_idx]
        wrist = joints[wrist_idx]
        
        # 计算手臂弯曲角度
        elbow_angle = self.get_limb_angle(joints, f'{side}_shoulder', f'{side}_elbow', f'{side}_wrist')
        
        # 判断手臂高度
        if wrist[1] > shoulder[1]:
            height_desc = "举高"
        elif wrist[1] < shoulder[1] - 0.3:
            height_desc = "放低"
        else:
            height_desc = "平举"
        
        # 判断弯曲程度
        if elbow_angle < 90:
            bend_desc = "紧弯"
        elif elbow_angle < 150:
            bend_desc = "弯曲"
        else:
            bend_desc = "伸直"
        
        side_cn = "左" if side == "left" else "右"
        return f"{side_cn}臂{height_desc}，{bend_desc}"

    def describe_leg_position(self, joints, side):
        """描述腿部位置"""
        hip_idx = self.joint_idx[f'{side}_hip']
        knee_idx = self.joint_idx[f'{side}_knee']
        ankle_idx = self.joint_idx[f'{side}_ankle']
        
        # 计算膝盖弯曲角度
        knee_angle = self.get_limb_angle(joints, f'{side}_hip', f'{side}_knee', f'{side}_ankle')
        
        # 判断弯曲程度
        if knee_angle < 90:
            bend_desc = "深蹲"
        elif knee_angle < 150:
            bend_desc = "弯曲"
        else:
            bend_desc = "伸直"
        
        # 判断腿部前后位置
        hip = joints[hip_idx]
        ankle = joints[ankle_idx]
        
        if ankle[2] > hip[2] + 0.2:
            position_desc = "前伸"
        elif ankle[2] < hip[2] - 0.2:
            position_desc = "后撤"
        else:
            position_desc = "自然站立"
        
        side_cn = "左" if side == "left" else "右"
        return f"{side_cn}腿{position_desc}，{bend_desc}"

    def describe_head_orientation(self, joints):
        """描述头部朝向"""
        neck_idx = self.joint_idx['neck']
        head_idx = self.joint_idx['head']
        
        neck = joints[neck_idx]
        head = joints[head_idx]
        
        head_vector = head - neck
        
        # 判断头部上下倾斜
        if head_vector[1] > 0.1:
            vertical_desc = "抬头"
        elif head_vector[1] < -0.1:
            vertical_desc = "低头"
        else:
            vertical_desc = "平视"
        
        # 判断头部左右转动
        if head_vector[0] > 0.1:
            horizontal_desc = "向右转"
        elif head_vector[0] < -0.1:
            horizontal_desc = "向左转"
        else:
            horizontal_desc = ""
        
        if horizontal_desc:
            return f"头部{vertical_desc}并{horizontal_desc}"
        else:
            return f"头部{vertical_desc}"

    def describe_body_posture(self, joints):
        """描述整体身体姿态"""
        pelvis_idx = self.joint_idx['pelvis']
        spine3_idx = self.joint_idx['spine3']
        
        pelvis = joints[pelvis_idx]
        spine3 = joints[spine3_idx]
        
        spine_vector = spine3 - pelvis
        
        # 判断身体前后倾斜
        if spine_vector[2] > 0.2:
            return "身体前倾"
        elif spine_vector[2] < -0.2:
            return "身体后仰"
        else:
            return "身体直立"

    def generate_description(self, joints):
        """
        生成关节点的完整文本描述
        
        Args:
            joints: numpy array, shape (22, 3), 关节坐标
            
        Returns:
            str: 姿态的文本描述
        """
        if joints.shape != (22, 3):
            raise ValueError("关节坐标应为 (22, 3) 的数组")
        
        descriptions = []
        
        # 描述整体姿态
        body_posture = self.describe_body_posture(joints)
        descriptions.append(body_posture)
        
        # 描述头部
        head_desc = self.describe_head_orientation(joints)
        descriptions.append(head_desc)
        
        # 描述双臂
        left_arm_desc = self.describe_arm_position(joints, 'left')
        right_arm_desc = self.describe_arm_position(joints, 'right')
        descriptions.extend([left_arm_desc, right_arm_desc])
        
        # 描述双腿
        left_leg_desc = self.describe_leg_position(joints, 'left')
        right_leg_desc = self.describe_leg_position(joints, 'right')
        descriptions.extend([left_leg_desc, right_leg_desc])
        
        return "，".join(descriptions) + "。"

    def get_joint_coordinates(self, joints, joint_name):
        """获取指定关节的坐标"""
        idx = self.joint_idx[joint_name]
        return joints[idx]

    def calculate_distance(self, joints, joint1_name, joint2_name):
        """计算两个关节间的距离"""
        idx1 = self.joint_idx[joint1_name]
        idx2 = self.joint_idx[joint2_name]
        return np.linalg.norm(joints[idx1] - joints[idx2])


# 使用示例
def demo():
    # 创建描述生成器
    generator = JointDescriptionGenerator()
    
    # 示例关节数据（随机生成的示例数据）
    # 实际使用时替换为你的真实关节坐标
    np.random.seed(42)
    sample_joints = np.random.randn(22, 3)
    
    # 调整一些关节位置使其更符合人体结构
    sample_joints[0] = [0, 0, 0]  # pelvis作为原点
    sample_joints[15] = [0, 1.5, 0]  # head在pelvis上方
    sample_joints[12] = [0, 1.2, 0]  # neck
    
    # 生成描述
    description = generator.generate_description(sample_joints)
    print("姿态描述:", description)
    
    # 获取特定关节信息
    head_pos = generator.get_joint_coordinates(sample_joints, 'head')
    print(f"头部坐标: {head_pos}")
    
    # 计算关节间距离
    arm_length = generator.calculate_distance(sample_joints, 'left_shoulder', 'left_wrist')
    print(f"左臂长度: {arm_length:.2f}")

if __name__ == "__main__":
    demo()


# 初始化生成器
generator = JointDescriptionGenerator()
joints=torch.load('motion_save.pt') #[num_rep, num_samples, 22, 3, n_frames]
joints=joints[0,0,:,:,0] 

# 生成描述
description = generator.generate_description(joints)
print(description)