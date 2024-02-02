import os
import google.generativeai as genai
from dotenv import load_dotenv
from crewai import Agent, Task, Process, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
# Load environment variables from .env
load_dotenv()
api_gemini = os.environ.get("GEMINI-API-KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.1, google_api_key=api_gemini
)

# Agent 1: Computer Vision Specialist
computer_vision_agent = Agent(
    role='Computer Vision Specialist',
    goal='Capture frames from a live camera feed using OpenCV and focus on facial and hand expressions.',
    backstory="""You specialize in computer vision and image processing. Your expertise lies in capturing and analyzing frames from live camera feeds. In this task, you'll be responsible for implementing OpenCV to capture frames and extracting relevant information regarding facial and hand expressions.""",
    verbose=True,
    allow_delegation=True,  # Set to True if collaboration is needed
    # Add any necessary tools or libraries for computer vision
    llm=llm # to load gemini
)

technologist = Agent(
    role="Technology Expert",
    goal="Make assessment on how technologically feasable the company is and what type of technologies the company needs to adopt in order to succeed",
    backstory="""You are a visionary in the realm of technology, with a deep understanding of both current and emerging technological trends. Your 
		expertise lies not just in knowing the technology but in foreseeing how it can be leveraged to solve real-world problems and drive business innovation.
		You have a knack for identifying which technological solutions best fit different business models and needs, ensuring that companies stay ahead of 
		the curve. Your insights are crucial in aligning technology with business strategies, ensuring that the technological adoption not only enhances 
		operational efficiency but also provides a competitive edge in the market.""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
    llm=llm # to load gemini
)

# Agent 2: Pose Estimation Expert
pose_estimation_agent = Agent(
    role='Pose Estimation Expert',
    goal='Integrate OpenPose for accurate pose estimation from the captured frames.',
    backstory="""Your expertise lies in pose estimation, and you're skilled at integrating OpenPose for precise analysis of human body poses. In this task, your role is to implement OpenPose and extract detailed information about facial and hand gestures from the captured frames.""",
    verbose=True,
    allow_delegation=True,  # Set to True if collaboration is needed
    # Add any necessary tools or libraries for pose estimation
    llm=llm # to load gemini

)

# Agent 3: 3D Graphics Developer
graphics_developer_agent = Agent(
    role='3D Graphics Developer',
    goal='Match the detected pose to a 3D model, emphasizing facial and hand expressions.',
    backstory="""You are proficient in 3D graphics development, and your task is to integrate the detected poses with a 3D model. Emphasis should be on accurately matching facial and hand expressions. You will need to recommend suitable 3D graphics libraries and handle the integration process.""",
    verbose=True,
    allow_delegation=True,  # Set to True if collaboration is needed
    # Add any necessary tools or libraries for 3D graphics development
    llm=llm # to load gemini
)

business_consultant = Agent(
    role="Business Development Consultant",
    goal="Evaluate and advise on the business model, scalability, and potential revenue streams to ensure long-term sustainability and profitability",
    backstory="""You are a seasoned professional with expertise in shaping business strategies. Your insight is essential for turning innovative ideas 
		into viable business models. You have a keen understanding of various industries and are adept at identifying and developing potential revenue streams. 
		Your experience in scalability ensures that a business can grow without compromising its values or operational efficiency. Your advice is not just
		about immediate gains but about building a resilient and adaptable business that can thrive in a changing market.""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
    llm=llm # to load gemini
)


# Task 1: Capture Frames using OpenCV
task_capture_frames = Task(
    description="""Guide me through the process of using OpenCV to capture frames from a live camera feed.""",
    agent=computer_vision_agent
)

# Task 2: Integrate OpenPose for Pose Estimation
task_pose_estimation = Task(
    description="""Integrate OpenPose for precise pose estimation, focusing on facial and hand expressions.""",
    agent=pose_estimation_agent
)

# Task 3: Match Detected Pose to a 3D Model
task_match_to_3d_model = Task(
    description="""Guide me on matching the detected pose to a 3D model, with a focus on facial and hand expressions. Provide recommendations for suitable 3D graphics libraries in Python.""",
    agent=graphics_developer_agent
)

task = Task(
    description="""Create a structure path and suggestion for achiving this task.
    `Guide me through the process of using OpenCV to capture frames from a live camera feed, integrating OpenPose for pose estimation, and finally, matching the detected pose to a 3D model. I'm particularly interested in focusing on facial and hand expressions, and I'd like advice on loading and manipulating 3D models in a Python environment. Can you provide step-by-step instructions and recommend suitable 3D graphics libraries for this task?`
    """,
    agent=technologist,
)

# Instantiate the crew with a sequential process
ai_team_crew = Crew(
    agents=[computer_vision_agent, pose_estimation_agent, graphics_developer_agent, business_consultant, technologist],
    tasks=[task],
    verbose=2,
    process=Process.sequential
)

# Get the AI team to work!
result = ai_team_crew.kickoff()

print("######################")
print(result)