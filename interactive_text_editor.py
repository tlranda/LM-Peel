import curses
import pdb

class text_trimmer():
    def __init__(self, text):
        self.text = text
        self.current_chunk = None
        self.cursor_pos = 0
        # Get terminal width
        try:
            self.max_width = curses.COLS
        except AttributeError:
            self.max_width = 80
        self.chunk_text()

    def __getitem__(self, key):
        if self.current_chunk != key:
            self.cursor_pos = 0
        self.current_chunk = key
        return self.chunks[key]

    def chunk_text(self, text=None):
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
            chunks.append(newchunk)
            chunk_idxes.append(idxes)
            marks.append([None] * len(idxes))
            terminators.append(termination)
        self.chunks = chunks
        self.chunk_idxes = chunk_idxes
        self.marks = marks
        self.current_chunk = 0
        self.terminators = terminators

    def build_chunk(self, words, lengths):
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
            # If there's a newline in this segment, it ends
            if '\n' in chunk[-idxes[-1]:]:
                # But we have to splice anything beyond it back on
                oldlen = len(chunk)
                splice_loc = chunk.index('\n')
                splice_segment = chunk[splice_loc+1:]
                chunk = chunk[:splice_loc]
                lengths.insert(0,len(splice_segment))
                words.insert(0,splice_segment)
                used -= len(chunk)-oldlen
                terminator = "\n"
                break
            # Add the space cost before looping
            used += 1
            if len(lengths) == 0 or used + lengths[0] >= self.max_width:
                break
        return chunk, idxes, words, lengths, terminator

    def move_cursor(self, direction, jump=None):
        now = self.chunk_idxes[self.current_chunk].index(self.cursor_pos)
        if direction == curses.KEY_LEFT:
            now -= 1
            if now < 0:
                return
        elif direction == curses.KEY_RIGHT:
            now += 1
            if now >= len(self.chunk_idxes[self.current_chunk]):
                return
        self.cursor_pos = self.chunk_idxes[self.current_chunk][now]
        if jump is not None and jump > 0:
            self.move_cursor(direction, jump-1)

    def mark(self):
        to_mark = self.chunk_idxes[self.current_chunk].index(self.cursor_pos)
        if self.marks[self.current_chunk][to_mark] is None:
            self.marks[self.current_chunk][to_mark] = 'X'
        else:
            self.marks[self.current_chunk][to_mark] = None

    def markstring(self):
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
                    pdb.set_trace()
                    raise
                first_half = (consider_len-1)//2
                if marks is None:
                    prefix = " "*max(0,first_half-1)+"^"
                    suffix = " "*(subchunk_len-len(prefix))
                    #out += " "*max(0,((subchunk_len-1)//2+(subchunk_len%2)-1))+"^"+" "*max(0,((subchunk_len-2)//2))
                else:
                    prefix = " "*max(0,first_half-1)+"V"
                    suffix = " "*(subchunk_len-len(prefix))
                    #out += " "*max(0,((subchunk_len-1)//2+(subchunk_len%2)-1))+"V"+" "*max(0,((subchunk_len-2)//2))
                out += prefix+suffix
            elif marks is None:
                out += " "*subchunk_len
            elif marks == 'X':
                out += "X"*subchunk_len
            sum_pos += subchunk_len
        return out

    def mask(self):
        out = ""
        for marks, chunk_idx, chunk, terminator in zip(self.marks, self.chunk_idxes, self.chunks, self.terminators):
            chunk_idx += [len(chunk)]
            for metaidx2, (mark, idx) in enumerate(zip(marks, chunk_idx)):
                if mark is None:
                    out += chunk[idx:chunk_idx[metaidx2+1]]
            out += terminator
        return out

def text_input_with_cursor(stdscr, text):
    # Disable cursor blinking
    curses.curs_set(0)
    chunker = text_trimmer(text)

    chunkidx = 0
    cursor_position = 0
    jump = None
    while True:
        stdscr.clear()
        # Display text
        try:
            if len(chunker[chunkidx]) == 0:
                chunkidx += 1
                continue
            stdscr.addstr(0, 0, chunker[chunkidx])
        except IndexError:
            break
        # Display cursor line under the current text
        stdscr.addstr(1, 0, chunker.markstring())
        stdscr.refresh()
        # Get user input
        key = stdscr.getch()
        if key in [curses.KEY_RIGHT, curses.KEY_LEFT]:
            chunker.move_cursor(key, jump)
            jump = None
        elif key in [ord(str(num)) for num in range(10)]:
            # Expect 2 to be +2 chunks, so -1 extra
            if jump is None:
                jump = int(key)-49
            else:
                jump *= 10
                jump += int(key)-49
        elif key == ord('x') or key == ord("X"):
            #curses.endwin()
            chunker.mark()
        elif key == 27: # ESC key
            chunkidx += 1
    return chunker

def main():
    # Sample text
    text = \
    """abc1 abc2 abc3 abc4 abc5 abc6 abc7 abc8 abc9 abc10 abc11 abc12 abc13
"""
    """This is a sample text. You can navigate through the words with arrow keys.
It is very very long and you may need more than an entire terminal window to parse all of the text within a singular line. That's a shame because I really want to ramble on and on about things.




Another line to gotcha with.
"""
    #chunker = text_trimmer(text)
    chunker = curses.wrapper(text_input_with_cursor, text)
    print(chunker.chunks)
    """
    import pdb
    pdb.set_trace()
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

