fn split_newest(stack: &mut [u8], frame_len: usize) -> (&mut [u8], &mut [u8]) {
    stack.split_at_mut(stack.len() - frame_len)
}

pub fn push_frame(stack: &mut [u8], frame_len: usize, render: impl FnOnce(&mut [u8])) {
    stack.copy_within(frame_len.., 0);
    let (_, newest) = split_newest(stack, frame_len);
    render(newest);
}

pub fn fill_frames(stack: &mut [u8], frame_len: usize, render: impl FnOnce(&mut [u8])) {
    let (older, newest) = split_newest(stack, frame_len);
    render(newest);
    for frame in older.chunks_exact_mut(frame_len) {
        frame.copy_from_slice(newest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_LEN: usize = 2;

    #[test]
    fn push_drops_the_oldest_frame_and_appends_the_newest() {
        let mut stack = [1, 1, 2, 2, 3, 3];
        push_frame(&mut stack, FRAME_LEN, |frame| frame.fill(4));
        assert_eq!(stack, [2, 2, 3, 3, 4, 4]);
    }

    #[test]
    fn fill_replaces_every_frame_with_the_rendered_one() {
        let mut stack = [1, 1, 2, 2, 3, 3];
        fill_frames(&mut stack, FRAME_LEN, |frame| frame.fill(9));
        assert_eq!(stack, [9; 6]);
    }

    #[test]
    fn single_frame_stack_holds_only_the_newest_frame() {
        let mut stack = [1, 1];
        push_frame(&mut stack, FRAME_LEN, |frame| frame.fill(2));
        assert_eq!(stack, [2, 2]);
    }
}
