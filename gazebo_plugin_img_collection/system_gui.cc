#include <functional>
#include <gazebo/gui/GuiIface.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/gazebo.hh>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Quaternion.hh>
#include <cmath>

namespace gazebo
{
  class SystemGUI : public SystemPlugin
  {
    /////////////////////////////////////////////
    /// \brief Destructor
    public: virtual ~SystemGUI()
    {
      this->connections.clear();
      if (this->userCam)
        this->userCam->EnableSaveFrame(false);
      this->userCam.reset();
    }

    /////////////////////////////////////////////
    /// \brief Called after the plugin has been constructed.
    public: void Load(int /*_argc*/, char ** /*_argv*/)
    {
      this->connections.push_back(
          event::Events::ConnectPreRender(
            std::bind(&SystemGUI::Update, this)));

      this->LoadCSV("test_trajectory_6dof_rpy_order_17th_may.csv"); //("training_data_6dof_14thmay.csv"); 
    }

    /////////////////////////////////////////////
    // \brief Called once after Load
    private: void Init()
    {
    }

    /////////////////////////////////////////////
    /// \brief Called every PreRender event. See the Load function.
    private: void Update()
    {
            
      if (this->currentPoseIndex >= this->poses.size())
      {
        std::cout << "Finished all poses.\n";
        this->userCam->EnableSaveFrame(false);
        return;
      }

      if (this->updateCounter < 100)
 	      {
    		    ++this->updateCounter;    			  
            std::cout << "count is less than 100\n";
            return;    		    	
  	    }
      
      // Get scene pointer
      rendering::ScenePtr scene = rendering::get_scene();

      // Wait until the scene is initialized.
      if (!scene || !scene->Initialized())
         return;       


      // // Look for a specific visual by name.
      // if (scene->GetVisual("ground_plane"))
      //   std::cout << "Has ground plane visual\n";
        
      this->userCam = gui::get_active_camera();
      if (!this->userCam)
      {         
        
        std::cout << "user Cam value is null \n";

      }

      else
      {
         std::cout << "user Camera is valid \n";
        
         // Enable saving frames
        this->userCam->EnableSaveFrame(true);

        // Specify the path to save frames into
        this->userCam->SetSaveFramePathname("/media/siva/vol1/gazebo_training_frames_17thmay");       //("test_path_frames_14thmay"); 

        // // NEW: Modify the camera pose
        // ignition::math::Pose3d newPose;
        // newPose.Pos() = ignition::math::Vector3d(0, 0, 1); // move camera
        // newPose.Rot() = ignition::math::Quaterniond(0, 0.3, 0); // tilt down slightly
        // this->userCam->SetWorldPose(newPose);

        // Set the camera pose
        ignition::math::Pose3d pose = this->poses[this->currentPoseIndex];
        this->userCam->SetWorldPose(pose);
        std::cout << "Set camera to pose index: " << this->currentPoseIndex << std::endl;

        this->currentPoseIndex++;

        double currentFOV = this->userCam->HFOV().Radian();
        ignition::math::Angle newFOV(2.0944);  // 90 degrees in radians
        this->userCam->SetHFOV(newFOV);
        std::cout << "Current horizontal FOV (radians): " << currentFOV << std::endl;


      }
    }

    private: void LoadCSV(const std::string &_filename)
    {
      std::ifstream file(_filename);
      if (!file.is_open())
      {
        std::cerr << "Failed to open file: " << _filename << std::endl;
        return;
      }

      std::string line;

      // Skip the header line
      if (!std::getline(file, line))
      {
        std::cerr << "CSV file is empty: " << _filename << std::endl;
        return;
      }

      while (std::getline(file, line))
      {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> values;
        
        while (std::getline(ss, value, ','))
        {
          values.push_back(std::stod(value));
        }

        if (values.size() != 6)
        {
          std::cerr << "Invalid row (expected 6 values): " << line << std::endl;
          continue;
        }

        // if (values.size() != 3)
        // {
        //   std::cerr << "Invalid row (expected 3 values): " << line << std::endl;
        //   continue;
        // }

        ignition::math::Vector3d position(values[0], values[1], values[2]);

        double yaw = values[3] * M_PI / 180.0;
        double pitch = values[4] * M_PI / 180.0;
        double roll = values[5] * M_PI / 180.0;

        // double roll = values[2] * M_PI / 180.0;
        // double pitch = 0;
        // double yaw = 0; //values[2] * M_PI / 180.0;

        //ignition::math::Quaterniond rotation(values[3], values[4], values[5]); // angles should be in degrees

        ignition::math::Quaterniond rotation(roll, pitch, yaw);
        ignition::math::Pose3d pose(position, rotation);

        this->poses.push_back(pose);
      }

      file.close();
      std::cout << "Loaded " << this->poses.size() << " poses from CSV.\n";
    }


    /// Pointer the user camera.
    private: rendering::UserCameraPtr userCam;

    /// All the event connections.
    private: std::vector<event::ConnectionPtr> connections;
    
    private: int updateCounter = 0;

    private: std::vector<ignition::math::Pose3d> poses;
    private: size_t currentPoseIndex = 0;
 

  };

  // Register this plugin with the simulator
  GZ_REGISTER_SYSTEM_PLUGIN(SystemGUI)
}
