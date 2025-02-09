import curses
import os
import pathlib
import subprocess

text_editor_bin = None
# Set default editor if environment variable is set
if 'EDITOR' in os.environ:
    text_editor_bin = os.environ['EDITOR']

def pick_text_editor():
    global text_editor_bin
    maybe_editors = ['vi','vim','neovim','emacs','nano','gedit']
    found = {}
    for editor in maybe_editors:
        resp = subprocess.run(['which',editor], capture_output=True)
        if len(resp.stdout) > 0:
            found[editor] = resp.stdout.decode().rstrip()
    ite = text_trimmer('\n'.join(found.keys()))
    ite.cursor_marktext(instructions="Mark ONE editor you'd like to use (or use the EDITOR environment variable in the future to skip)")
    key = ite.mask(invert=True).rstrip()
    text_editor_bin = found[key]

def edit_via_editor(text, editor=None, tmp=None, unlink=True):
    global text_editor_bin
    if tmp is None:
        tmp = pathlib.Path('tmp_editing.txt')
    tmp = pathlib.Path(tmp)
    version_number = 0
    while tmp.exists():
        tmp = tmp.with_stem(f"{tmp.stem}_{version_number}")
        version_number += 1
    with open(tmp,'w') as f:
        f.write("".join(text))
    if editor is None:
        if text_editor_bin is None:
            pick_text_editor()
        editor = text_editor_bin
    subprocess.call([editor, tmp])
    with open(tmp,'r') as f:
        new_text = "".join(f.readlines())
    if unlink:
        tmp.unlink()
    return new_text

def chunker_with_cursor(stdscr, chunker, instructions=None):
    # Disable cursor blinking
    curses.curs_set(0)
    if not chunker.max_width_OK:
        # Now that we have an environment, should be OK to reset
        chunker._reset_max_width()
        # Have to re-assign chunks post-reset for it to really work
        chunker.chunk_text()

    editor_instructions = "Arrow/hjkl keys to move; 'x' to mark indicated word, 'X' to mark indicated line, 'q', 'Q' or ESCAPE key to confirm and exit"
    ending = "<<END OF MESSAGE>>"
    # This function keeps track of # lines displayed, not the chunker
    max_lines = curses.LINES
    if instructions is not None:
        # Remove a line from the budget
        max_lines -= 1
    jump = None
    chunkidx = 0 # Currently edited line, tracked SEPARATELY from object
    while True:
        stdscr.clear()
        # Pin instructions to the top of the screen
        stdscr.addstr(0,0,editor_instructions+("" if jump is None else f" JUMP: {jump+1}"), curses.A_REVERSE)
        DISPLAYED_LINES = 1
        if instructions is not None:
            stdscr.addstr(DISPLAYED_LINES, 0, instructions, curses.A_REVERSE)
            DISPLAYED_LINES += 1
        # Fix chunkidx indexing errors that may occur due to jumps
        if chunkidx >= chunker.nchunks:
            chunkidx = chunker.nchunks-2
        elif chunkidx < 0:
            chunkidx = 0

        # Display text
        # Budget to display in top-half: (max_lines-3)//2
        for _ in range(max(0,chunkidx-((max_lines-3)//2)),chunkidx):
            # Use .getchunk to not mess with .__getitem__ metadata
            stdscr.addstr(DISPLAYED_LINES, 0, chunker.getchunk(_))
            DISPLAYED_LINES += 1
        # Autoskip empty lines to a line worth editing
        if len(chunker[chunkidx]) == 0:
            chunkidx += 1
            continue
        stdscr.addstr(DISPLAYED_LINES, 0, chunker[chunkidx])
        # Display cursor line under the current text
        stdscr.addstr(DISPLAYED_LINES+1, 0, chunker.markstring())
        DISPLAYED_LINES += 2
        # Budget to display: Up until DISPLAYED_LINES == max_lines
        suffix_chunk = chunkidx+1
        while suffix_chunk < chunker.nchunks and DISPLAYED_LINES < max_lines-1:
            stdscr.addstr(DISPLAYED_LINES, 0, chunker.getchunk(suffix_chunk))
            suffix_chunk += 1
            DISPLAYED_LINES += 1
        if suffix_chunk == chunker.nchunks:
            # Show ending
            stdscr.addstr(DISPLAYED_LINES, 0, ending, curses.A_REVERSE)
        stdscr.refresh()

        # Get user input
        key = stdscr.getch()
        if key in [curses.KEY_RIGHT, curses.KEY_LEFT, ord('h'), ord('l')]:
            # Vim keys rebind to arrow keys so I don't have to push logic into
            # move_cursor()
            if key == ord('h'):
                key = curses.KEY_LEFT
            elif key == ord('l'):
                key = curses.KEY_RIGHT
            move = chunker.move_cursor(key, jump)
            # Move indicates the chunker said we need to change lines as a response
            # to the cursor movement
            if move is not None:
                chunkidx += move
                # Skip backwards past empty lines until we hit an editable one
                while chunkidx > 0 and move < 0 and chunker.getchunk(chunkidx) == '':
                    chunkidx -= 1
                if chunkidx < 0:
                    chunkidx = 0
            # Clear jumps after moving
            jump = None
        elif key in [curses.KEY_UP, curses.KEY_DOWN, ord('j'), ord('k')]:
            # Vim keys rebind to arrow keys so I don't have to push logic into
            # move_cursor()
            if key == ord('j'):
                key = curses.KEY_DOWN
            elif key == ord('k'):
                key = curses.KEY_UP
            if key == curses.KEY_UP:
                # Extra +1's here to meet what the user probably thought 1-based indexing wise
                if jump is not None:
                    chunkidx -= jump+1
                else:
                    chunkidx -= 1
                while chunkidx > 0 and chunker.getchunk(chunkidx) == '':
                    chunkidx -= 1
            else:
                if jump is not None:
                    chunkidx += jump+1
                else:
                    chunkidx += 1
            # Clear jumps after moving
            jump = None
            # Reset cursor position on the new line -- we don't know if the old position will be valid on this new line or not
            chunker.cursor_position = 0
        elif key in [ord(str(num)) for num in range(10)]:
            # Expect 2 to be +2 chunks, so -1 extra
            if jump is None:
                jump = int(key)-49
            else:
                jump += 1
                jump *= 10
                jump += int(key)-49
        elif key == ord('x'):
            chunker.mark()
        elif key == ord("X"):
            chunker.mark_chunk()
        elif key == 5: # CTRL+e, retry editing via editor function
            return edit_via_editor(chunker.text)
        elif key in [ord('Q'), ord('q'), 27]: # 27 == ESC key
            break

class text_trimmer():
    """
        This class helps you visually delete sections of some sprawling text,
        ie a very minimal, deletion-only text editor.
        It supports a limited amount of conveniences and is mostly there to help
        proofread and reduce verbosity from things like LLM responses that have
        a parseable subset/substring that may be otherwise difficult to parse.
    """
    def __init__(self, text):
        """
            The text to parse is given at initialization time, and automatically
            attempt to slice and dice it up for editing. While you can
            technically re-use this object by re-assigning and re-chunking
            text, this isn't tested or a very supported use case as of yet.
        """
        self.text = text
        self.current_chunk = None
        self.cursor_pos = 0
        # Get terminal width to determine line breaks so that the indicator line
        # works as expected
        try:
            self.max_width = curses.COLS
            self.max_width_OK = True
        except AttributeError:
            self.max_width = 80
            self.max_width_OK = False
        self.chunk_text()

    def _reset_max_width(self):
        """
            USUALLY you should not need to call this function, it should be used
            when the max width changes but I do not intend to really support that
            to a high degree. The most likely valid use case is that you didn't
            have a curses screen set up when the object was created and now that
            you do have it, the curses.COLS attribute won't error.
            That's why I am not try/catching the attribute request here -- if it
            is an exception, you probably should not be calling this function in
            the first place!
        """
        self.max_width = curses.COLS
        self.max_width_OK = True
        self.chunk_text()

    def chunk_text(self, text=None):
        """
            Figure out how to separate the text into iterable line-by-line chunks
            based on the screen width we can display, then build associated
            mapping based on space delimiters to accept/reject words.
        """
        if text is None:
            text = self.text
        words = text.split(' ')
        lengths = list(map(len,words))
        chunks = []
        chunk_idxes = []
        marks = []
        terminators = []
        while len(words) > 0:
            newchunk, idxes, words, lengths, termination = self.build_chunk(words, lengths)
            if len(terminators) > 0 and terminators[-1] == '\n' and termination == '\n' and len(newchunk) == 0:
                # Grow multiline whitespace in the previous terminator rather
                # than having to click through allllll of the whitespace
                terminators[-1] += '\n'
                continue
            chunks.append(newchunk)
            chunk_idxes.append(idxes)
            # Marks are None (no effect) by default
            marks.append([None] * len(idxes))
            terminators.append(termination)
        # Update all relevant object memory to reflect newly assigned data
        self.chunks = chunks
        self.nchunks = len(chunks)
        self.chunk_idxes = chunk_idxes
        self.marks = marks
        self.current_chunk = 0
        self.terminators = terminators

    def build_chunk(self, words, lengths):
        """
            Consume up to our width out of the words/lengths here, returning
            any unused portion to be used on subsequent calls
        """
        used = 0
        chunk = ""
        idxes = [0]
        terminator = ""
        while True:
            if used > 0:
                # Only add the space/idx on second-loop beyond
                chunk += " "
                idxes.append(used)
            used += lengths.pop(0)
            chunk += words.pop(0)
            # If there's a newline in this segment, it ends at the newline
            if '\n' in chunk[-idxes[-1]:]:
                terminator = "\n"
                # But we have to splice anything beyond the newline back on so
                # the next iteration doesn't delete it
                oldlen = len(chunk)
                # First index only, we handle multiline whitespace in the
                # calling function
                splice_loc = chunk.index('\n')
                splice_segment = chunk[splice_loc+1:]
                chunk = chunk[:splice_loc]
                # Add whatever came afterwards
                lengths.insert(0,len(splice_segment))
                words.insert(0,splice_segment)
                # I don't think this matters but it doesn't hurt to have it
                used -= len(chunk)-oldlen
                break
            # Add the space cost before looping
            used += 1
            if len(lengths) == 0 or used + lengths[0] >= self.max_width:
                break
        return chunk, idxes, words, lengths, terminator

    def __getitem__(self, key):
        """
            Allow array-like indexing that has SIDE EFFECTS of changing the
            current chunk. If you do not want the side effect, use .getchunk()
            instead.
        """
        if self.current_chunk != key:
            self.cursor_pos = 0
        self.current_chunk = key
        return self.chunks[key]

    def getchunk(self, idx):
        """
            Fech a chunk by index without updating cursor position/current chunk
        """
        return self.chunks[idx]


    def move_cursor(self, direction, jump=None):
        """
            Push the cursor in indicated direction, jump parameter indicates
            a multi-unit move if it is Integer-typed rather than None
        """
        now = self.chunk_idxes[self.current_chunk].index(self.cursor_pos)
        if direction == curses.KEY_LEFT:
            now -= 1
            if now < 0:
                # Signal to caller that we need to wrap around to previous line
                return -1
        elif direction == curses.KEY_RIGHT:
            now += 1
            if now >= len(self.chunk_idxes[self.current_chunk]):
                # Signal to caller that we need to wrap around to next line
                return 1
        # Otherwise push yourself in the indicated direction
        self.cursor_pos = self.chunk_idxes[self.current_chunk][now]
        # Just recursively do this -- the base cases above will not let you
        # left/right multiple lines but it's not a big deal to support that
        if jump is not None and jump > 0:
            self.move_cursor(direction, jump-1)

    def mark(self, value=None):
        """
            Edit the mark value at the current position. For now I'm only
            considering a binary Marked-As-Value/Not Marked but it may be nice
            to have other mark types too in the future
        """
        if value is None:
            value = 'X'
        if type(value) is not str or len(value) > 1:
            raise ValueError("Mark strings must be 1-length!")
        to_mark = self.chunk_idxes[self.current_chunk].index(self.cursor_pos)
        # Mark
        if self.marks[self.current_chunk][to_mark] is None:
            self.marks[self.current_chunk][to_mark] = value
        # Unmark
        elif self.marks[self.current_chunk][to_mark] == value:
            self.marks[self.current_chunk][to_mark] = None
        # Supercede mark with new mark
        else:
            self.marks[self.current_chunk][to_mark] = value

    def mark_chunk(self, value=None):
        """
            Edit the mark values at every position on the line without pushing
            the cursor anywhere
            Same marking rationale as .mark()
        """
        if value is None:
            value = 'X'
        for idx in range(len(self.marks[self.current_chunk])):
            # Mark
            if self.marks[self.current_chunk][idx] is None:
                self.marks[self.current_chunk][idx] = value
            # Unmark
            elif self.marks[self.current_chunk][idx] == value:
                self.marks[self.current_chunk][idx] = None
            # Supercede mark with new mark
            else:
                self.marks[self.current_chunk][idx] = value

    def markstring(self):
        """
            Return a line-length string displaying marks assigned to word-chunks
        """
        chunk = self.chunks[self.current_chunk]
        idxes = self.chunk_idxes[self.current_chunk] + [len(chunk)]
        sum_pos = 0
        out = ""
        for metaidx, (marks, chunk_idx) in enumerate(zip(self.marks[self.current_chunk], idxes)):
            subchunk_len = idxes[metaidx+1]-chunk_idx
            if self.cursor_pos == sum_pos:
                try:
                    if metaidx < len(idxes)-2 and chunk[idxes[metaidx+1]] == ' ':
                        consider_len = subchunk_len - 1
                    else:
                        consider_len = subchunk_len
                except:
                    curses.endwin()
                    raise
                first_half = (consider_len-1)//2
                if marks is None:
                    prefix = " "*max(0,first_half-1)+"^"
                    suffix = " "*(subchunk_len-len(prefix))
                    #out += " "*max(0,((subchunk_len-1)//2+(subchunk_len%2)-1))+"^"+" "*max(0,((subchunk_len-2)//2))
                else:
                    prefix = " "*max(0,first_half-1)+marks
                    suffix = " "*(subchunk_len-len(prefix))
                    #out += " "*max(0,((subchunk_len-1)//2+(subchunk_len%2)-1))+"V"+" "*max(0,((subchunk_len-2)//2))
                out += prefix+suffix
            elif marks is None:
                out += " "*subchunk_len
            else:
                out += marks*subchunk_len
            sum_pos += subchunk_len
        return out

    def mask(self, invert=False):
        """
            Primary use case for postprocessing, reconstruct the chunked string
            to its original form, minus any portions marked as "X"

            Invert to ONLY include portions marked as "X"
        """
        out = ""
        for marks, chunk_idx, chunk, terminator in zip(self.marks, self.chunk_idxes, self.chunks, self.terminators):
            chunk_idx += [len(chunk)]
            anything_written = False
            for metaidx2, (mark, idx) in enumerate(zip(marks, chunk_idx)):
                # We still have some wonky interactions with multiline whitespace
                # but this helps us get rid of it entirely if the line before it
                # is completely empty (or if the line itself is)
                if (mark is None and not invert) or (mark == 'X' and invert):
                    new_out = chunk[idx:chunk_idx[metaidx2+1]]
                    out += new_out
                    anything_written = len(new_out) > 0
                elif (mark == 'X' and not invert) or (mark is None and invert):
                    continue
                else:
                    raise ValueError(f"No masking behavior implemented for mark '{mark}'")
            if anything_written:
                out += terminator
        return out

    def cursor_marktext(self, instructions = None):
        """
            Run your own cursor across the text and handle everything, the
            callback here doesn't use the object directly but maybe it could,
            I do not care to address that right now

            Instructions passed as extra prompt beyond editor controls
        """
        return curses.wrapper(chunker_with_cursor, self, instructions)

# This function can be used as a callback to CREATE the trimmer object with
# given text and then duplicate chunker_with_cursor -- it was for development
# and probably should be destroyed
def text_input_with_cursor(stdscr, text):
    # Disable cursor blinking
    curses.curs_set(0)
    chunker = text_trimmer(text)

    jump = None
    instructions = "Arrow keys to move; 'x' to delete indicated word, 'X' to delete indicated line, 'Q' or ESCAPE key to confirm and exit"
    ending = "<<END OF MESSAGE>>"
    max_lines = curses.LINES
    chunkidx = 0
    cursor_position = 0
    while True:
        stdscr.clear()
        stdscr.addstr(0,0,instructions+("" if jump is None else f" JUMP: {jump+1}"), curses.A_REVERSE)
        DISPLAYED_LINES = 1
        if chunkidx >= chunker.nchunks:
            chunkidx = chunker.nchunks-2
        elif chunkidx < 0:
            chunkidx = 0
        # Display text
        # Budget to display: (max_lines-3)//2
        for _ in range(max(0,chunkidx-((max_lines-3)//2)),chunkidx):
            # Use .getchunk to not mess with .__getitem__ metadata
            stdscr.addstr(DISPLAYED_LINES, 0, chunker.getchunk(_))
            DISPLAYED_LINES += 1
        if len(chunker[chunkidx]) == 0:
            chunkidx += 1
            continue
        stdscr.addstr(DISPLAYED_LINES, 0, chunker[chunkidx])
        # Display cursor line under the current text
        stdscr.addstr(DISPLAYED_LINES+1, 0, chunker.markstring())
        DISPLAYED_LINES += 2
        # Budget to display: Up until DISPLAYED_LINES == max_lines
        suffix_chunk = chunkidx+1
        while suffix_chunk < chunker.nchunks and DISPLAYED_LINES < max_lines-1:
        #for _ in range(chunkidx+1, min(chunkidx+1+((max_lines-3)//2),chunker.nchunks)):
            stdscr.addstr(DISPLAYED_LINES, 0, chunker.getchunk(suffix_chunk))
            suffix_chunk += 1
            DISPLAYED_LINES += 1
        if suffix_chunk == chunker.nchunks:
            # Show ending
            stdscr.addstr(DISPLAYED_LINES, 0, ending, curses.A_REVERSE)
        stdscr.refresh()
        # Get user input
        key = stdscr.getch()
        if key in [curses.KEY_RIGHT, curses.KEY_LEFT]:
            move = chunker.move_cursor(key, jump)
            if move is not None:
                chunkidx += move
                while chunkidx > 0 and move < 0 and chunker.getchunk(chunkidx) == '':
                    chunkidx -= 1
                if chunkidx < 0:
                    chunkidx = 0
            jump = None
        elif key in [curses.KEY_UP, curses.KEY_DOWN]:
            if key == curses.KEY_UP:
                if jump is not None:
                    chunkidx -= jump
                else:
                    chunkidx -= 1
                while chunkidx > 0 and chunker.getchunk(chunkidx) == '':
                    chunkidx -= 1
            else:
                if jump is not None:
                    chunkidx += jump
                else:
                    chunkidx += 1
            jump = None
            chunker.cursor_position = 0
        elif key in [ord(str(num)) for num in range(10)]:
            # Expect 2 to be +2 chunks, so -1 extra
            if jump is None:
                jump = int(key)-49
            else:
                jump += 1
                jump *= 10
                jump += int(key)-49
        elif key == ord('x'):
            chunker.mark()
        elif key == ord("X"):
            chunker.mark_chunk()
        elif key in [ord('Q'), ord('q'), 27]: # 27 == ESC key
            break
    return chunker

def main():
    # Sample text
    text = \
    """This is a sample text. You can navigate through the words with arrow keys.
It is very very long and you may need more than an entire terminal window to parse all of the text within a singular line. That's a shame because I really want to ramble on and on about things.




Another line to gotcha with.
To invert the foreground and background colors of a line of text in Python using the curses library, you can use the curses.A_REVERSE attribute. This attribute will swap the foreground and background colors for the text it's applied to.
This will display a line of text with the foreground and background colors inverted, and another line with a custom color pair (blue background with white text).
Number of double-precision floating-point multiply-accumulate operations executed by non-predicated threads. Each multiply-accumulate operation contributes 1 to the count.
The dependency and wait time analysis between different threads and CUDA streams only takes into account execution dependencies stated in the respective supported API contracts. This especially does not include synchronization as a result of resource contention. For example, asynchronous memory copies enqueued into independent CUDA streams will not be marked dependent even if the concrete GPU has only a single copy engine. Furthermore, the analysis does not account for synchronization using a not-supported API. For example, a CPU thread actively polling for a value at some memory location (busy-waiting) will not be considered blocked on another concurrent activity.
The dependency analysis has only limited support for applications using CUDA Dynamic Parallelism (CDP). CDP kernels can use CUDA API calls from the GPU which are not tracked via the CUPTI Activity API. Therefore, the analysis cannot determine the full dependencies and waiting time for CDP kernels. However, it utilizes the parent-child launch dependencies between CDP kernels. As a result the critical path will always include the last CDP kernel of each host-launched kernel.
This section contains detailed descriptions of the metrics that can be collected by nvprof and the Visual Profiler. A scope value of “Single-context” indicates that the metric can only be accurately collected when a single context (CUDA or graphic) is executing on the GPU. A scope value of “Multi-context” indicates that the metric can be accurately collected when multiple contexts are executing on the GPU. A scope value of “Device” indicates that the metric will be collected at device level, that is it will include values for all the contexts executing on the GPU. Note that, NVLink metrics collected for kernel mode exhibit the behavior of “Single-context”.
Stalled for not selected - Warp was ready but did not get a chance to issue as some other warp was selected for issue. This reason generally indicates that kernel is possibly optimized well but in some cases, you may be able to decrease occupancy without impacting latency hiding, and doing so may help improve cache hit rates.
NVIDIA Nsight Compute is an interactive kernel profiler for CUDA applications. It provides detailed performance metrics and API debugging via a user interface and command line tool. In addition, its baseline feature allows users to compare results within the tool. Nsight Compute provides a customizable and data-driven user interface and metric collection and can be extended with analysis scripts for post-processing results. Refer to the nvprof Transition Guide section in the Nsight Compute CLI document. Refer to the Visual Profiler Transition Guide section in the Nsight Compute document.
A security vulnerability issue required profiling tools to disable features using GPU performance counters for non-root or non-admin users when using a Windows 419.17 or Linux 418.43 or later driver. By default, NVIDIA drivers require elevated permissions to access GPU performance counters. On Tegra platforms, profile as root or using sudo. On other platforms, you can either start profiling as root or using sudo, or by enabling non-admin profiling. More details about the issue and the solutions can be found on the ERR_NVGPUCTRPERM web page.
OpenACC profiling might fail when OpenACC library is linked statically in the user application. This happens due to the missing definition of the OpenACC API routines needed for the OpenACC profiling, as compiler might ignore definitions for the functions not used in the application. This issue can be mitigated by linking the OpenACC library dynamically.
Profiling features for devices with compute capability 7.5 and higher are supported in the NVIDIA Nsight Compute. Visual Profiler does not support Guided Analysis, some stages under Unguided Analysis and events and metrics collection for devices with compute capability 7.5 and higher. One can launch the NVIDIA Nsight Compute UI for devices with compute capability 7.5 and higher from Visual Profiler. Also nvprof does not support query and collection of events and metrics, source level analysis and other options used for profiling on devices with compute capability 7.5 and higher. The NVIDIA Nsight Compute command line interface can be used for these features.
THIS DOCUMENT AND ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE. TO THE EXTENT NOT PROHIBITED BY LAW, IN NO EVENT WILL NVIDIA BE LIABLE FOR ANY DAMAGES, INCLUDING WITHOUT LIMITATION ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, PUNITIVE, OR CONSEQUENTIAL DAMAGES, HOWEVER CAUSED AND REGARDLESS OF THE THEORY OF LIABILITY, ARISING OUT OF ANY USE OF THIS DOCUMENT, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. Notwithstanding any damages that customer might incur for any reason whatsoever, NVIDIA’s aggregate and cumulative liability towards customer for the products described herein shall be limited in accordance with the Terms of Sale for the product.
"""
    """abc1 abc2 abc3 abc4 abc5 abc6 abc7 abc8 abc9 abc10 abc11 abc12 abc13
"""
    chunker = text_trimmer(text)
    chunker.cursor_marktext()
    #chunker = curses.wrapper(text_input_with_cursor, text)
    print(chunker.chunks)
    """
    print(chunker[0])
    print(chunker.markstring())
    chunker.move_cursor(curses.KEY_RIGHT)
    print(chunker[0])
    print(chunker.markstring())
    chunker.mark()
    chunker.move_cursor(curses.KEY_RIGHT)
    print(chunker[0])
    print(chunker.markstring())
    """
    print("\n".join([str(_) for _ in chunker.marks]))
    print(chunker.mask())

if __name__ == '__main__':
    main()

