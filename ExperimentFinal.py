import psychopy
psychopy.useVersion('2023.1.3')

# we import the relevant libaries and modules
from psychopy import visual, core, event, gui
import random
import csv
import os
import datetime
import tempfile
from PIL import Image, ImageFilter
import numpy as np

# we set up the dialog box to get participant info
class ParticipantInfo:
    def __init__(self):
        dlg = gui.Dlg(title="Please enter a four-digit Participant ID")
        dlg.addField('Participant ID:')
        ok_data = dlg.show()
        if not dlg.OK or not ok_data[0]:
            core.quit()
        self.participant_id = ok_data[0]

# this is the overall class for the experiment
class MaskedFaceRecognitionExperiment:
    def __init__(self, face_dir="C:/Users/winth/Documents/KDV/EM2/Face_recognition_experiment/Greyscale ansigterne/frontal_faces_adjusted_greyscaled_background_adjusted",
                 num_trials=40, num_blocks=6, num_test_trials=40): 
        self.participant_info = ParticipantInfo()
        self.win = visual.Window(fullscr=True, color="white")
        self.instructions = visual.TextStim(self.win, text='', color='black', wrapWidth=1.5)
        self.results = []
        self.test_results = []
        self.num_trials = num_trials # Nu 40 pr. blok
        self.num_blocks = num_blocks
        self.num_test_trials = num_test_trials
        self.face_dir = face_dir

        # we find all the image files in our face directory
        all_files = [os.path.join(face_dir, f) for f in os.listdir(face_dir) if f.lower().endswith((".jpg"))]
        random.shuffle(all_files)  #bland rækkefølgen

        total_needed = int(num_blocks * num_trials * 4.5 + num_test_trials * 4.5) # we compute the total number of images needed
        if float(len(all_files)) < float(total_needed):
            raise ValueError(f"Ikke nok billeder i mappen! Kræver {total_needed}, men fandt {len(all_files)}.")
        self.available_faces = all_files[:] #appending all images to available faces
        self.used_faces = [] # list to keep track of used faces
        self.temp_dir = tempfile.TemporaryDirectory()
        self._temp_files = [] # a list to store temporary filenames created for blurred/pixelated images - the experiment bugs if not saved for cleanup later

        # we set up the response keys
        keys_list = ['x', 'm']
        self.yes_key, self.no_key = keys_list

        # we set up the different mask types with grades/intensities
        self.mask_types_with_grades = [
            # Mouth Masked
            "mouth_masked_1", "mouth_masked_2", "mouth_masked_3", 
            # Blurred
            "blurred_1", "blurred_2", "blurred_3", 
            # Gabor
            "gabor_1", "gabor_2", "gabor_3",
            # Unmasked
            "unmasked" 
        ]
        # we set up a list of only the maskings - unmasked is not included
        self.graded_mask_types = [
            "mouth_masked_1", "mouth_masked_2", "mouth_masked_3",
            "blurred_1", "blurred_2", "blurred_3",
            "gabor_1", "gabor_2", "gabor_3",
        ]
    # we set up a function that shows instructions in the experiment
    def show_instructions(self, text):
        self.instructions.text = text
        self.instructions.draw()
        self.win.flip()
        event.waitKeys()

    # we set up a function that draws a face with a given mask type on the center of the screen. This function is responsible for displaying the image
    def draw_face(self, face_path, mask_type):
        self.draw_face_no_flip(face_path, mask_type, size=(0.5, 0.7), pos=(0, 0))
        self.win.flip()
        # remove temporary files only after flip so the driver/GPU has the file available
        self._cleanup_temp_files()

    # this function apllies the mask type to the face and draws it to the back buffer without displaying it yet. It bugged when trting to flip in the same function.
    def draw_face_no_flip(self, face_path, mask_type, size=(0.5, 0.7), pos=(0,0)):
        # no masking(unmasked)
        if mask_type == "unmasked":
            stim = visual.ImageStim(self.win, image=face_path, size=size, pos=pos)
            stim.draw()
            return

        # We split mask_type and intensity grade
        if "_" in mask_type:
            base, grade = mask_type.rsplit("_", 1)
            try:
                grade = int(grade)
            except ValueError:
                base, grade = mask_type, 1
        else:
            base, grade = mask_type, 1
        
        # makes sure only intensities 1, 2, 3 are used
        if grade not in [1, 2, 3]:
            raise ValueError(f"Ugyldig maskegrade: {grade} for {mask_type}. Skal være 1, 2 eller 3.")


        # we set up the intensities for each mask type. Every parameter needed for the mask types is considered and graded in lists.

        # Mouth Masked
        if base == "mouth_masked":
            stim = visual.ImageStim(self.win, image=face_path, size=size, pos=pos)
            stim.draw()
            #variable  heights for the mask depending on grade
            mask_heights = [0.08, 0.18, 0.30]
            mask_height = size[1] * mask_heights[grade-1]
            mask = visual.Rect(
                self.win,
                width=size[0],
                height=mask_height,
                fillColor='black',
                lineColor='black',
                pos=(pos[0], pos[1] - size[1] * 0.22)
            )
            mask.draw()
        # Blurred:
        elif base == "blurred":
            img = Image.open(face_path).convert("RGB")
            #variable blur radii depending on grade
            blur_radii = [2, 8, 16] 
            blur_radius = blur_radii[grade-1]
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            img.save(tmp.name)
            stim = visual.ImageStim(self.win, image=tmp.name, size=size, pos=pos)
            stim.draw()
            self._temp_files.append(tmp.name)
        # Gabor
        elif base == "gabor":
            img = Image.open(face_path)
            arr = np.array(img, dtype=float) / 255.0
            h, w = arr.shape[:2]
            # variable cycles and blend factors depending on grade
            n_cycles_list = [4, 8, 12]
            blend_factors = [0.15, 0.4, 0.6] 
            
            n_cycles = n_cycles_list[grade-1]
            blend = blend_factors[grade-1]
              
            x = np.linspace(0, np.pi * 2 * n_cycles, w)
            gabor_pattern = 0.5 + 0.5 * np.sin(x)
            xx = np.linspace(-1, 1, w)
            gauss = np.exp(- (xx**4) * 1)
            gabor_pattern *= gauss
            gabor = np.tile(gabor_pattern, (h, 1))
            if arr.ndim == 3:
                gabor = np.repeat(gabor[:, :, np.newaxis], 3, axis=2)
            blended = arr * (1 - blend) + gabor * blend
            blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
            blended_img = Image.fromarray(blended)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            blended_img.save(tmp.name)
            stim = visual.ImageStim(self.win, image=tmp.name, size=size, pos=pos)
            stim.draw()
            self._temp_files.append(tmp.name)
        else:
            raise ValueError(f"Ugyldig mask_type: {mask_type}")

    # we clean up the temporary files created for the blurred and gabor masked conditions.
    def _cleanup_temp_files(self):
        for fp in getattr(self, "_temp_files", [])[:]:
            try:
                os.unlink(fp)
            except Exception:
                pass
        self._temp_files = []
    #the function that runs a single trial of the experiment
    def run_trial(self, trial_num, memory_mask_type, probe_mask_type, save_results=True, probe_in_set=None):
        if len(self.available_faces) < 5:
            self.show_instructions("Ikke flere ansigter til rådighed! Eksperimentet stoppes.")
            self.win.close()
            core.quit()
            return False

        # we select 4 faces for the memory set
        memory_set = random.sample(self.available_faces, 4)
        for face in memory_set:
            self.available_faces.remove(face) #the faces are removed from available faces
            self.used_faces.append(face) #and added to used faces

        # we select the probe face
        if probe_in_set is None:
            probe_in_set = random.choice([True, False])

        if probe_in_set: # ==TRUE
            probe = random.choice(memory_set) #we choose one of the memory set faces as probe
        else:
            probe_candidates = [f for f in self.available_faces if f not in memory_set]
            if not probe_candidates:
                self.show_instructions("Ikke flere ansigter til rådighed til probe! Eksperimentet stoppes.")
                self.win.close()
                core.quit()
                return False
            probe = random.choice(probe_candidates) #we choose a face not in the memory set as probe
            self.available_faces.remove(probe)
            self.used_faces.append(probe)
        # we set the positions and size for the memory set faces on the window
        positions = [(-0.45, 0.35), (0.45, 0.35), (-0.45, -0.35), (0.45, -0.35)]
        mem_size = (0.35, 0.5)

        # we draw each face in the memory set with the specified mask type
        for face, pos in zip(memory_set, positions):
            try:
                # Test at billedet kan åbnes
                Image.open(face)
                self.draw_face_no_flip(face, memory_mask_type, size=mem_size, pos=pos)
            except Exception as e:
                print(f"Fejl ved åbning/draw af {face}: {e}")

        # we flip to show the memory set
        self.win.flip()
        # we clean up any temporary files created for the masks
        self._cleanup_temp_files()
        memory_exposure = 2.5  #seconds
        core.wait(memory_exposure)
        
        # clear screen after memory exposure
        visual.TextStim(self.win, text=" ", color="white").draw()
        self.win.flip()
        core.wait(0.05) #a brief interstimulus interval

        # we show a fixation cross for a variable time before the probe
        fixation = visual.TextStim(self.win, text="+", color="black")
        fixation.draw()
        self.win.flip()
        v = 1.0  # rate parameter 
        fixation_time = np.clip(np.random.normal(loc=v, scale=0.15), 0.7, 1.4)  # between 0.7 og 1.4 seconds
        core.wait(fixation_time)

        # we show the probe face with the specified mask type
        self.draw_face_no_flip(probe, probe_mask_type, size=(0.5, 0.7), pos=(0, 0))
        self.win.flip()
        self._cleanup_temp_files()

        # Response collection
        timer = core.Clock()
        keys = event.waitKeys(keyList=[self.yes_key, self.no_key, 'escape'], timeStamped=timer)
        response, rt = 'none', None
        if keys:  # we check if any key was pressed
            key, time = keys[0]
            if key == 'escape':
                self.win.close()
                core.quit()
                return False
            elif key == self.yes_key:
                response = 'yes'
                rt = time
            elif key == self.no_key:
                response = 'no'
                rt = time

        # Normalize reaction time to seconds if some systems return milliseconds
        if rt is not None:
            if rt > 50:
                rt = rt / 1000.0
            rt = round(rt, 3)

        if response == 'escape':
            self.win.close()
            core.quit()
            return False

        # show blank screen for 1 second before next trial
        visual.TextStim(self.win, text=" ", color="white").draw()
        self.win.flip()
        core.wait(1)

        correct = (probe_in_set and response == 'yes') or (not probe_in_set and response == 'no') #we determine if the response was correct or not

        #we save the trial data
        data_row = [
            self.participant_info.participant_id,
            trial_num,
            memory_mask_type,
            probe_mask_type,
            probe_in_set,
            response,
            correct,
            rt,
        ]
        if save_results:
            self.results.append(data_row)
        else:
            self.test_results.append(data_row)

        return True

    # we set up the training block function
    def training_block(self):
        """Training block built the same way as main_run but uses num_test_trials and does NOT save results."""
        self.show_instructions(
            "You will firstly do a training block, that will look exactly the same as the main experiment.\n\n"
            "In every trial you will be presented with four faces, followed by a fixation-cross, followed by a target. \n\n"
            f"Press '{self.yes_key}' if the target-face was one of the four faces just presented to you. \n\n"
            f"Press '{self.no_key}' if the target-face was NOT one of the four faces just presented to you. \n\n"
            "Please answer as ACCURATELY and QUICKLY as possible.\n\n"
            "Press any key to begin the training block"
        )
        mask_types = self.graded_mask_types

        full_factorial = [] #we set up a list containing all combinations of trial types, mask types and probe_in_set conditions
        for trial_type in ('A', 'B'): #memory masked or probe masked
            for mask in mask_types:
                for probe_in in (True, False):
                    if trial_type == 'A':
                        memory_mask = "unmasked"
                        probe_mask = mask
                    else:
                        memory_mask = mask
                        probe_mask = "unmasked"
                    full_factorial.append((memory_mask, probe_mask, probe_in))

        # we add baseline trials (4 unmasked trials in total)
        unmasked_base = [("unmasked", "unmasked", True), ("unmasked", "unmasked", False)]
        unmasked_base *= 2
        
        full_factorial.extend(unmasked_base) # in total: 36 + 4 = 40 trials
        
        n_needed = self.num_test_trials

        base = full_factorial[:] # 40 trials
        trial_pool = []
        
        # if num_test_trials is less than 40, we sample without replacement to get the needed number of trials. for the training block we only need 10
        reps = n_needed // len(base)
        remainder = n_needed % len(base)
        trial_pool.extend(base * reps)
        if remainder:
            trial_pool.extend(random.sample(base, remainder))
        
        if n_needed == len(base):
            trial_pool = base

        random.shuffle(trial_pool) #we randomize the order of trials in a block

        # we run each trial in the training block
        for t, (memory_mask, probe_mask, probe_in_set) in enumerate(trial_pool, start=1):
            cont = self.run_trial(trial_num=f"training_trial{t}",
                                  memory_mask_type=memory_mask,
                                  probe_mask_type=probe_mask,
                                  save_results=False,
                                  probe_in_set=probe_in_set)
            if not cont:
                return False

        self.show_instructions("The training block is over. Press any key to start the main experiment.")
        return True

    # we set up the main experiment function. it follows the same structure as the training block but saves results and uses num_trials and num_blocks
    def main_run(self):
        mask_types = self.graded_mask_types 

        trial_idx = 0
        for block in range(self.num_blocks):
            full_factorial = []
            for trial_type in ('A', 'B'):
                for mask in mask_types:
                    for probe_in in (True, False):
                        if trial_type == 'A':
                            memory_mask = "unmasked"   
                            probe_mask = mask
                        else:
                            memory_mask = mask
                            probe_mask = "unmasked"   
                        full_factorial.append((memory_mask, probe_mask, probe_in))

            unmasked_base = [("unmasked", "unmasked", True), ("unmasked", "unmasked", False)]
            unmasked_base *= 2
            
            full_factorial.extend(unmasked_base) 

            n_needed = self.num_trials 
            base = full_factorial[:]
            trial_pool = []

            reps = n_needed // len(base)
            remainder = n_needed % len(base)
            trial_pool.extend(base * reps)
            if remainder:
                trial_pool.extend(random.sample(base, remainder))
            
            if n_needed == len(base):
                trial_pool = base

            random.shuffle(trial_pool) 

            for t, (memory_mask, probe_mask, probe_in_set) in enumerate(trial_pool, start=1):
                cont = self.run_trial(trial_num=f"block{block+1}_trial{t}",
                                      memory_mask_type=memory_mask,
                                      probe_mask_type=probe_mask,
                                      probe_in_set=probe_in_set)
                if not cont:
                    return False
                trial_idx += 1

            self.show_instructions(f"Block {block+1} out of {self.num_blocks} is done. Please take a break if needed. \n\n"
                                   f"REMEMBER: '{self.yes_key.upper()}' = YES, '{self.no_key.upper()}' = NO\n\n"
                                   "Press any key to continue with the main experiment.")
        return True

    #we set up the function that saves the results to a CSV file
    def save_results(self):
        filename = f"ID_{self.participant_info.participant_id}.csv"
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file, delimiter=',')
            # Updated header to include both memory and probe mask types
            writer.writerow(["participant_id", "trial", "memory_mask", "probe_mask", "probe_in_set", "response", "correct", "rt"])
            writer.writerows(self.results)

    # we set up the survey function that runs at the end of the experiment
    def run_survey(self):
        """
        At end of experiment ask participant to rate how well each mask_type masked the face (1-9).
        Ratings are appended to self.results so they end up in the same CSV file.
        """
        # we randomize the order of mask types presented in the survey
        mask_types = self.graded_mask_types
        random.shuffle(mask_types)

        self.show_instructions(
            "Lastly we would like for you to answer a short survey. You will be shown each mask again one at a time.\n\n"
            "For each mask, please rate how well you think it hides the face on a scale from 1 (not at all) to 9 (very well).\n\n"
            "Press a number key (1-9) to submit your rating.\n\n"
            "Please take your time and press any key to begin the survey."
        )

        # we choose a sample face to display with each mask type during the survey
        if self.available_faces:
            sample = self.available_faces[0]
        elif self.used_faces:
            sample = self.used_faces[0]
        else:
            sample = None

        for mask in mask_types:
            if sample:
                self.draw_face_no_flip(sample, mask, size=(0.5, 0.7), pos=(0, 0.3))

            prompt = visual.TextStim(
                self.win,
                text=(
                    "How well does this mask type hide the face?\n\n"
                    "Press BACKSPACE to clear your choice. Press ENTER to confirm your rating."
                ),
                color='black',
                pos=(0, -0.35),
                wrapWidth=1.8
            )

            rating_display = visual.TextStim(
                self.win,
                text="Current rating: _",
                color='black',
                pos=(0, -0.65)
            )

            current_rating = None
            confirmed = False
            timer = core.Clock()

            while not confirmed:
                if sample:
                    self.draw_face_no_flip(sample, mask, size=(0.5, 0.7), pos=(0, 0.3))
                prompt.draw()
                rating_display.text = f"Current rating: {current_rating if current_rating else '_'}"
                rating_display.draw()
                self.win.flip()
                self._cleanup_temp_files()

                keys = event.waitKeys(
                    keyList=[str(n) for n in range(1, 10)] + ['return', 'escape', 'backspace', 'delete'],
                    timeStamped=timer
                )

                if not keys:
                    continue

                key, t = keys[0]

                if key == 'escape':
                    self.win.close()
                    core.quit()
                    return False

                elif key in [str(n) for n in range(1, 10)]:
                    current_rating = int(key)

                elif key in ['backspace', 'delete']:
                    current_rating = None

                elif key == 'return' and current_rating is not None:
                    confirmed = True
                    rating = current_rating
                    rt = round(t, 3)
            # we save the survey data
            data_row = [
                self.participant_info.participant_id,
                f"survey_{mask}",
                mask,
                "",
                "",
                rating,
                "",
                rt,
            ]
            self.results.append(data_row)

            visual.TextStim(self.win, text=" ", color="white").draw()
            self.win.flip()
            core.wait(0.3)

        return True
    # we set up the main function that runs the experiment
    def run(self):
        print(f"Antal billeder: {len([os.path.join(self.face_dir, f) for f in os.listdir(self.face_dir) if f.lower().endswith('.jpg')])}")
        print(f"Ja-tast er {self.yes_key} og Nej-tast er {self.no_key}")
        print(f"Starter eksperiment med {self.num_blocks} blocks og {self.num_trials} trials pr. block")
        self.show_instructions(
            "Welcome to our facial recognition experiment.\n\n"
            "In this experiment you will be presented with a memory set of four faces. Afterwards another face (a target).\n"
            "The faces we present to you may or may not be masked/degraded in a particular way.\n\n"
            "Your task is to identify if the target was part of the memory set or not.\n\n"
            "The experiment is divided into a training block and a main experiment. Data will ONLY be collected from the main experiment.\n"
            f"The main experiment is divided into {self.num_blocks} blocks. \n\n"
            "Press any key to start the experiment"
        )
        self.training_block()
        self.main_run()

        # survey at end of experiment
        self.run_survey()

        self.save_results()
        self.show_instructions("Thank you for participating in the experiment!\nWe really appreciate you taking the time to complete it. \n\nPress any key to exit the window.")
        self.win.close()
        core.quit()
    
    def cleanup(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()
        print("Oprydning af midlertidig mappe fuldført.")

# we set up the main execution block
if __name__ == "__main__":
    exp = None
    try:
        exp = MaskedFaceRecognitionExperiment(
            face_dir="C:/Users/winth/Documents/KDV/EM2/Face_recognition_experiment/Greyscale ansigterne/frontal_faces_adjusted_greyscaled_background_adjusted",
            num_trials= 40, # 40 trials pr. blok
            num_blocks= 6, # 6 blokke i alt
            num_test_trials= 10 # 10 trials i træningsblokken
        )
        exp.run()
    except Exception as e:
        print("Fejl under eksperiment:", e)
        import traceback; traceback.print_exc()
    finally:
        if exp:
            exp.cleanup()