import json
import numpy as np
import time
from typing import ClassVar, Optional

from vision_assist.models import Path, Instruction, FinalAnswer

# TODO:
# - sort out the calculations for determining danger level
# - using event or processing time for previous instruction key vals?
# - include previous instructions in instruction calculations?
# - implement _analyse_paths?
# - implement _analyse_previous_instructions?

class PathAnalyser:
    _instance: ClassVar[Optional['PathAnalyser']] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Ensure only one instance of PathAnalyser exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the path analyser only once."""
        if not self._initialized:
            self._initialized = True
            self.paths: list[Path] = []
            
            # Convert timestamp to milliseconds (int) for dictionary keys
            self.previous_instructions: dict[int, list[Instruction]] = {}
            self.instructions: list[Instruction] = []

    def _analyse_path(self, path: Path) -> Instruction | None:
        """
        Analyse the properties of a path and determine if an instruction is necessary

        Args:
            path (Path): A path object

        Returns:
            Instruction: a single instruction
        """
        angle = path.angle
        length = path.length
        
        # determine a threshold for how much the angle needs to be in a certain length
        # we tell the user, if the path moves say 5 grids over 30 grids in length it is 
        # probably not necessary to inform the user but if it is more significant, we should
        
        # Determine if angle change is significant enough to warrant an instruction
        if length < (self.frame_height * 0.3):
            return None
            
        # Calculate danger based on angle and length
        if abs(angle) > 45:
            danger = "high"
        elif abs(angle) > 25:
            danger = "medium"
        else:
            danger = "low"
            
        # Determine instruction type based on angle change
        instruction_type = "bearing" if angle < 20 else "curve" if angle < 35 else "turn"
        direction = "straight" if path.start.x == path.end.x else "left" if path.start.x > path.end.x else "right"
            
        return Instruction(
            direction=direction,
            danger=danger,
            distance=length,
            start=path.start,
            end=path.end,
            angle_change=angle,
            length=length,
            instruction_type=instruction_type
        )
        
    def _analyse_corners(self, path: Path) -> list[Instruction]:
        """
        Analyse the properties of the corner and determine an instruction for the user

        Args:
            path (Path): A path object

        Returns:
            list[Instruction]: A list of instructions for the application to understand
        """
        instructions = []
        
        for corner in path.corners:
            # Higher y value means closer to bottom of frame/user
            distance = corner.start.y
            
            if distance < self.frame_height * 0.5:
                continue
            
            # TODO: determine whether to keep this logic?
            # Calculate position-based weight for angle importance
            # At bottom of frame (max y), weight is 0.3
            # At top of frame (min y), weight is 0.7
            # angle_weight = 0.3 + (0.4 * (1 - distance / self.frame_height))
            # height_weight = 1 - angle_weight
            
            # Use exponential functions for height and angle multipliers
            height_multiplier = np.exp((np.log(2) / self.frame_height) * distance) - 1
            # absolute value of angle change to prevent negative values
            angle_multiplier = np.exp((np.log(2) / 90) * abs(corner.angle_change)) - 1 
                        
            # TODO: determine whether to keep this logic?
            # danger_value = height_multiplier * height_weight + angle_multiplier * angle_weight
            danger_value = (height_multiplier * 0.7) + (angle_multiplier * 0.3)
            
            # TODO: determine whether to keep this logic?
            # # Near bottom of frame, only consider very sharp angles
            # if distance > (self.frame_height * 0.7) and abs(corner.angle_change) < 30:
            #     danger_value *= 0.5  # Reduce danger for shallow angles near bottom
            
            match danger_value:
                case value if value > 0.75:
                    danger = "immediate"
                case value if value > 0.65:
                    danger = "high"
                case value if value > 0.45:
                    danger = "medium"
                case _:
                    danger = "low"
        
            # Create instruction
            instruction = Instruction(
                direction=corner.direction,
                danger=danger,
                distance=distance,
                start=corner.start,
                end=corner.end,
                angle_change=corner.angle_change,
                length=corner.length,
                instruction_type="turn" if corner.sharpness == "sharp" else "curve"
            )
            
            instructions.append(instruction)
            
        return instructions
        
    def _analyse_instructions(self, instructions: list[Instruction]) -> list[Instruction]:
        """
        Analyse the instructions and determine if some are redundant
        
        Args:
            instructions (list[Instruction]): A list of instructions

        Returns:
            list[Instruction]: A list of instructions for the application to understand
        """
        # TODO: implement
        return instructions
    
    def _analyse_previous_instructions(
        self, 
        previous_instructions: dict[int, list[Instruction]], 
        current_instructions: list[Instruction],
        current_timestamp: int
    ) -> list[Instruction]:
        """
        Analyse the previous instructions and use it to enrich the current instruction.

        Args:
            previous_instructions (list[Instruction])
            current_instruction (list[Instruction])

        Returns:
            list[Instruction]: An updated list of instructions, enriched by previous instructions
        """
        danger_mapping = {
            "immediate": 0,
            "high": 1,
            "medium": 2,
            "low": 3
        }

        # Analyse the previous instructions sent
        if not previous_instructions:
            return current_instructions
        
        # Try to pair each previous instruction with a current instruction
        # There can be a many to one relationship between previous and current instructions
        pairs = []
        # 1.5s is the maximum time we will wait for an instruction to be paired, assuming we run at 2fps
        acceptable_time_difference = 1500
        
        for previous_timestamp, previous_instruction_list in previous_instructions.items():
            for previous_instruction in previous_instruction_list:
                for current_instruction in current_instructions:
                    # We should only ever pair a bearing with a bearing
                    if previous_instruction.instruction_type == "bearing" and current_instruction.instruction_type != "bearing":
                        continue
                    
                    # If the previous instruction is closer than the current instruction, skip
                    if previous_instruction.distance > current_instruction.distance:
                        continue
                    
                    # If the previous instruction is in a different direction, skip
                    if previous_instruction.direction != current_instruction.direction:
                        continue
                    
                    time_difference = current_timestamp - previous_timestamp
                    
                    y_difference = abs(previous_instruction.start.y - current_instruction.start.y)
                    # Adjust for the fact that the closer to the bottom of the frame we are, the more lenient we are
                    y_multiplier = (previous_instruction.start.y / self.frame_height)
                    # If the time difference is too much, or we have moved too much vertically,
                    # then we should not pair them together
                    y_acceptable = time_difference < acceptable_time_difference and (y_difference * y_multiplier) < self.frame_height * 0.2 
                    if not y_acceptable: continue
                    
                    x_difference = abs(previous_instruction.start.x - current_instruction.start.x)
                    # Adjust for the fact that the closer to the bottom of the frame we are, the more lenient we are
                    # Still use the y value as the user is expected to move in the y direction
                    x_multiplier = (previous_instruction.start.y / self.frame_height)
                    # If the time difference is too much, or we have moved too much horizontally,
                    x_acceptable = time_difference < acceptable_time_difference and (x_difference * x_multiplier) < self.frame_width * 0.2 
                    if not x_acceptable: continue
                    
                    # If the danger level of the previous instruction is greater than the current instruction,
                    # Then we should not pair them together because the danger level has gone down, howvever, if the danger
                    # level has stayed the same, or increased, then we should pair them together
                    if danger_mapping[previous_instruction.danger] - danger_mapping[current_instruction.danger] > 0:
                        continue
                    
                    pairs.append((previous_instruction, current_instruction, {"time_difference": time_difference, "y_difference": y_difference, "x_difference": x_difference}))


        # Now we can interate over every pair and determine if the current instruction should be enriched
        for previous_instruction, current_instruction, metadata in pairs:
            time_difference = metadata["time_difference"]
            y_difference = metadata["y_difference"]
            x_difference = metadata["x_difference"]
            direction_change = abs(previous_instruction.angle_change - current_instruction.angle_change)
            
            # If the current instruction is a bearing, we should only enrich it if there is a significant change in direction,
            # don't care about the time difference or distance in this case
            if current_instruction.instruction_type == "bearing":
                match current_instruction.danger:
                    case "high":
                        if direction_change > 12.5:
                            current_instruction.danger = "immediate"
                            print(f"Bearing danger increased to immediate: {current_instruction.model_dump()}")
                    case "medium":
                        if direction_change > 7.5:
                            current_instruction.danger = "high"
                            print(f"Bearing danger increased to high: {current_instruction.model_dump()}")
                    case "low":
                        if direction_change > 3.75:
                            current_instruction.danger = "medium"
                            print(f"Bearing danger increased to medium: {current_instruction.model_dump()}")
                    case _:
                        pass    
            else:
                match current_instruction.danger:
                    case "high":
                        if direction_change > 15:
                            current_instruction.danger = "immediate"
                            print(f"Bearing danger increased to immediate: {current_instruction.model_dump()}")
                    case "medium":
                        if direction_change > 10:
                            current_instruction.danger = "high"
                            print(f"Bearing danger increased to high: {current_instruction.model_dump()}")
                    case "low":
                        if direction_change > 7.5:
                            current_instruction.danger = "medium"
                            print(f"Bearing danger increased to medium: {current_instruction.model_dump()}")        
                    case _:
                        pass
            
        # Now remove any instructions which are too far away, or low in danger, DO NOT REMOVE BEARINGS
        for instruction in current_instructions:
            if instruction.instruction_type != "bearing":
                if instruction.danger == "low":
                    current_instructions.remove(instruction)
                # If they are in the top 33% of the frame, we should remove them
                elif instruction.distance < self.frame_height * 0.33:
                    current_instructions.remove(instruction)

        return current_instructions
    
    def determine_final_instruction(self, instructions: list[Instruction]) -> FinalAnswer:
        """
        Process analyzed instructions to return a single FinalAnswer enum value.
        Prioritizes immediate dangers, then uses most critical instruction.
        """    
        if not instructions:
            return FinalAnswer.CONTINUE_FORWARD
                
        # Check for immediate danger instructions first
        immediate_instructions = [i for i in instructions if i.danger == "immediate"]
        if immediate_instructions:
            direction = immediate_instructions[0].direction
            return FinalAnswer.MOVE_LEFT if direction == "left" else FinalAnswer.MOVE_RIGHT
        
        # Handle case with only a single bearing instruction
        if len(instructions) == 1 and instructions[0].instruction_type == "bearing":
            return FinalAnswer.CONTINUE_FORWARD
        
        # Get highest priority instruction based on sorting from PathAnalyser
        primary_instruction = instructions[0]
        
        # Map instruction directions to FinalAnswer
        if primary_instruction.direction == "left":
            return FinalAnswer.MOVE_LEFT
        elif primary_instruction.direction == "right":
            return FinalAnswer.MOVE_RIGHT
        
        return FinalAnswer.CONTINUE_FORWARD


    def __call__(self, frame_height: int, frame_width: int, paths: list[Path]) -> FinalAnswer:

        """
        Process and analyse the paths.
        
        Args:
            frame_height (int): The height of the frame
            paths (list[Path]): A list of paths

        Returns:
            list[Instruction]: A list of instructions for the application to understand
        """
        self.paths = paths
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        self.instructions = []
        
        # Convert to milliseconds for integer key
        processing_time = int(time.time() * 1000)
        
        for path in self.paths:
            path_instruction = self._analyse_path(path)
            if path_instruction: self.instructions.append(path_instruction)
            
            if path.corners:
                corner_instructions = self._analyse_corners(path)
                self.instructions.extend(corner_instructions)
                
        # Remove instructions that are redundant
        self.instructions = self._analyse_instructions(self.instructions)
        
        # Sort first by instruction type (bearing last), then by danger level
        def sort_key(instruction):
            # turn and curve are of same importance
            type_order = {'turn': 0, 'curve': 0, 'bearing': 1} 
            danger_order = {'immediate': 0, 'high': 1, 'medium': 2, 'low': 3}
            
            return (
                type_order[instruction.instruction_type],
                danger_order[instruction.danger]
            )
            
        self.unfiltered_instructions = sorted(self.instructions, key=sort_key)
        # print("Instructions before analysis with previous instructions:")
        # print(json.dumps([instruction.model_dump() for instruction in self.instructions], indent=4))
        
        self.filtered_instructions = self._analyse_previous_instructions(self.previous_instructions, self.instructions, processing_time)
        # print(f"Instructions after analysis with previous instructions: {self.filtered_instructions}\n")
        # print(json.dumps([instruction.model_dump() for instruction in self.filtered_instructions], indent=4))
        
        # print(f"Previous instructions were:")
        # for timestamp, instructions in self.previous_instructions.items():
        #     print(f"Timestamp: {timestamp}")
        #     print(json.dumps([instruction.model_dump() for instruction in instructions], indent=4))

        # TODO: process this in relation to previous instructions?
        # say, if we have three frames in a row where the instruction is turn right and the 
        # danger is increasing, we should definitely instruct the user to move
        self.previous_instructions[processing_time] = self.unfiltered_instructions
        
        # Remove any instructions from more than 5s ago
        self.previous_instructions = {
            timestamp: instructions 
            for timestamp, instructions in self.previous_instructions.items()
            if processing_time - timestamp <= 5000
        }
        
        final_answer = self.determine_final_instruction(self.filtered_instructions)
        
        return final_answer.value


# Create singleton for export
path_analyser = PathAnalyser()