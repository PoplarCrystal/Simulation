import pinocchio as pin
import numpy as np


class PinoSolver:
    def __init__(self, xml_path, xml_type="URDF"):
        if xml_type == "MJCF":
            self.model = pin.buildModelFromMJCF(xml_path)
        else:
            self.model = pin.buildGeomFromUrdf(xml_path)
        self.data = self.model.createData()
    
    def update_kin_dyn(self, q_pos: np.ndarray, q_vel: np.ndarray = None):
        pin.forwardKinematics(self.model, self.data, q_pos)
        pin.computeJointJacobians(self.model, self.data, q_pos)
        pin.updateFramePlacements(self.model, self.data)
        if q_vel is not None:
            pin.computeAllTerms(self.model, self.data, q_pos, q_vel)
        
    def get_frame_pose(self, frame_id):
        # return self.data.oMf[frame_id].homogeneous.copy()
        return self.data.oMf[frame_id]

    def get_frame_jac(self, frame_id, reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        return pin.getFrameJacobian(self.model, self.data, frame_id, reference_frame)
        
    def get_inertia_mat(self, q):
        return pin.crba(self.model, self.data, q).copy()

    def get_coriolis_mat(self, q, qdot):
        return pin.computeCoriolisMatrix(self.model, self.data, q, qdot).copy()

    def get_gravity_mat(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()
