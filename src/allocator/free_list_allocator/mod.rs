#![deny(unsafe_code, clippy::unwrap_used)]

#[cfg(feature = "visualizer")]
pub(crate) mod visualizer;

use super::{resolve_backtrace, AllocationReport, AllocationType, SubAllocator, SubAllocatorBase};
use crate::{AllocationError, Result};

use log::{log, Level};
use slotmap::{Key as _, SlotMap};
use std::collections::HashSet;

const USE_BEST_FIT: bool = true;

fn align_down(val: u64, alignment: u64) -> u64 {
    val & !(alignment - 1u64)
}

fn align_up(val: u64, alignment: u64) -> u64 {
    align_down(val + alignment - 1u64, alignment)
}

//pub type MemoryChunkId = std::num::NonZeroU64;
pub type MemoryChunkId = slotmap::DefaultKey;

#[derive(Debug)]
pub(crate) struct MemoryChunk {
    pub(crate) chunk_id: MemoryChunkId,
    pub(crate) size: u64,
    pub(crate) offset: u64,
    pub(crate) allocation_type: AllocationType,
    pub(crate) name: Option<String>,
    pub(crate) backtrace: Option<backtrace::Backtrace>, // Only used if STORE_STACK_TRACES is true
    next: MemoryChunkId,
    prev: MemoryChunkId,
}

#[derive(Debug)]
pub(crate) struct FreeListAllocator {
    size: u64,
    allocated: u64,
    pub(crate) chunks: SlotMap<MemoryChunkId, MemoryChunk>,
    free_chunks: HashSet<MemoryChunkId>,
}

/// Test if two suballocations will overlap the same page.
fn is_on_same_page(offset_a: u64, size_a: u64, offset_b: u64, page_size: u64) -> bool {
    let end_a = offset_a + size_a - 1;
    let end_page_a = align_down(end_a, page_size);
    let start_b = offset_b;
    let start_page_b = align_down(start_b, page_size);

    end_page_a == start_page_b
}

/// Test if two allocation types will be conflicting or not.
fn has_granularity_conflict(type0: AllocationType, type1: AllocationType) -> bool {
    if type0 == AllocationType::Free || type1 == AllocationType::Free {
        return false;
    }

    type0 != type1
}

impl FreeListAllocator {
    pub(crate) fn new(size: u64) -> Self {
        let mut chunks = SlotMap::default();

        let initial_chunk_id = chunks.insert_with_key(|initial_chunk_id| MemoryChunk {
            chunk_id: initial_chunk_id,
            size,
            offset: 0,
            allocation_type: AllocationType::Free,
            name: None,
            backtrace: None,
            prev: MemoryChunkId::null(),
            next: MemoryChunkId::null(),
        });

        let mut free_chunks = HashSet::default();
        free_chunks.insert(initial_chunk_id);

        Self {
            size,
            allocated: 0,
            chunks,
            free_chunks,
        }
    }

    /// Finds the specified `chunk_id` in the list of free chunks and removes if from the list
    fn remove_id_from_free_list(&mut self, chunk_id: MemoryChunkId) {
        self.free_chunks.remove(&chunk_id);
    }
    /// Merges two adjacent chunks. Right chunk will be merged into the left chunk
    fn merge_free_chunks(
        &mut self,
        chunk_left: MemoryChunkId,
        chunk_right: MemoryChunkId,
    ) -> Result<()> {
        // Gather data from right chunk and remove it
        let (right_size, right_next) = {
            let chunk = self.chunks.remove(chunk_right).ok_or_else(|| {
                AllocationError::Internal("Chunk ID not present in chunk list.".into())
            })?;
            self.remove_id_from_free_list(chunk.chunk_id);

            (chunk.size, chunk.next)
        };

        // Merge into left chunk
        {
            let chunk = self.chunks.get_mut(chunk_left).ok_or_else(|| {
                AllocationError::Internal("Chunk ID not present in chunk list.".into())
            })?;
            chunk.next = right_next;
            chunk.size += right_size;
        }

        // Patch pointers
        if !right_next.is_null() {
            let chunk = self.chunks.get_mut(right_next).ok_or_else(|| {
                AllocationError::Internal("Chunk ID not present in chunk list.".into())
            })?;
            chunk.prev = chunk_left;
        }

        Ok(())
    }
}

impl SubAllocatorBase for FreeListAllocator {}
impl SubAllocator for FreeListAllocator {
    #[allow(unsafe_code)]
    fn allocate(
        &mut self,
        size: u64,
        alignment: u64,
        allocation_type: AllocationType,
        granularity: u64,
        name: &str,
        backtrace: Option<backtrace::Backtrace>,
    ) -> Result<(u64, std::num::NonZeroU64)> {
        let free_size = self.size - self.allocated;
        if size > free_size {
            return Err(AllocationError::OutOfMemory);
        }

        let mut best_fit_id: MemoryChunkId = MemoryChunkId::null();
        let mut best_offset = 0u64;
        let mut best_aligned_size = 0u64;
        let mut best_chunk_size = 0u64;

        for &current_chunk_id in self.free_chunks.iter() {
            assert!(!current_chunk_id.is_null());

            let current_chunk = self.chunks.get(current_chunk_id).ok_or_else(|| {
                AllocationError::Internal(
                    "Chunk ID in free list is not present in chunk list.".into(),
                )
            })?;

            if current_chunk.size < size {
                continue;
            }

            let mut offset = align_up(current_chunk.offset, alignment);

            if !current_chunk.prev.is_null() {
                let previous = self.chunks.get(current_chunk.prev).ok_or_else(|| {
                    AllocationError::Internal("Invalid previous chunk reference.".into())
                })?;
                if is_on_same_page(previous.offset, previous.size, offset, granularity)
                    && has_granularity_conflict(previous.allocation_type, allocation_type)
                {
                    offset = align_up(offset, granularity);
                }
            }

            let padding = offset - current_chunk.offset;
            let aligned_size = padding + size;

            if aligned_size > current_chunk.size {
                continue;
            }

            if !current_chunk.next.is_null() {
                let next = self.chunks.get(current_chunk.next).ok_or_else(|| {
                    AllocationError::Internal("Invalid next chunk reference.".into())
                })?;
                if is_on_same_page(offset, size, next.offset, granularity)
                    && has_granularity_conflict(allocation_type, next.allocation_type)
                {
                    continue;
                }
            }

            if USE_BEST_FIT {
                if best_fit_id.is_null() || current_chunk.size < best_chunk_size {
                    best_fit_id = current_chunk_id;
                    best_aligned_size = aligned_size;
                    best_offset = offset;

                    best_chunk_size = current_chunk.size;
                };
            } else {
                best_fit_id = current_chunk_id;
                best_aligned_size = aligned_size;
                best_offset = offset;

                best_chunk_size = current_chunk.size;
                break;
            }
        }

        let first_fit_id = if !best_fit_id.is_null() {
            best_fit_id
        } else {
            return Result::Err(AllocationError::OutOfMemory);
        };

        let chunk_id = if best_chunk_size > best_aligned_size {
            let new_chunk_id;
            let prev_chunk;
            {
                let free_chunk = self.chunks.get(first_fit_id).ok_or_else(|| {
                    AllocationError::Internal("Chunk ID must be in chunk list.".into())
                })?;

                prev_chunk = free_chunk.prev;
                let offset = free_chunk.offset;

                new_chunk_id = self.chunks.insert_with_key(|new_chunk_id| MemoryChunk {
                    chunk_id: new_chunk_id,
                    size: best_aligned_size,
                    offset,
                    allocation_type,
                    name: Some(name.to_string()),
                    backtrace,
                    prev: prev_chunk,
                    next: first_fit_id,
                });

                // Safety: just a few lines above we request the entry via a checked `get`,
                // and return an error if it doesn't exist.
                let free_chunk = unsafe { self.chunks.get_unchecked_mut(first_fit_id) };

                free_chunk.prev = new_chunk_id;
                free_chunk.offset += best_aligned_size;
                free_chunk.size -= best_aligned_size;
            };

            if !prev_chunk.is_null() {
                let prev_chunk = self.chunks.get_mut(prev_chunk).ok_or_else(|| {
                    AllocationError::Internal("Invalid previous chunk reference.".into())
                })?;
                prev_chunk.next = new_chunk_id;
            }

            //println!("Chunks now at {} items", self.chunks.len());

            new_chunk_id
        } else {
            let chunk = self
                .chunks
                .get_mut(first_fit_id)
                .ok_or_else(|| AllocationError::Internal("Invalid chunk reference.".into()))?;

            chunk.allocation_type = allocation_type;
            chunk.name = Some(name.to_string());
            chunk.backtrace = backtrace;

            self.remove_id_from_free_list(first_fit_id);

            first_fit_id
        };

        self.allocated += best_aligned_size;

        Ok((
            best_offset,
            std::num::NonZeroU64::new(chunk_id.data().as_ffi())
                .expect("slotmap KeyData::as_ffi must be non-zero"),
        ))
    }

    fn free(&mut self, chunk_id: Option<std::num::NonZeroU64>) -> Result<()> {
        let chunk_id = chunk_id
            .ok_or_else(|| AllocationError::Internal("Chunk ID must be a valid value.".into()))?;

        let chunk_id = MemoryChunkId::from(slotmap::KeyData::from_ffi(chunk_id.get()));

        let (next_id, prev_id) = {
            let chunk = self.chunks.get_mut(chunk_id).ok_or_else(|| {
                AllocationError::Internal(
                    "Attempting to free chunk that is not in chunk list.".into(),
                )
            })?;
            chunk.allocation_type = AllocationType::Free;
            chunk.name = None;
            chunk.backtrace = None;

            self.allocated -= chunk.size;

            self.free_chunks.insert(chunk.chunk_id);

            (chunk.next, chunk.prev)
        };

        if !next_id.is_null() && self.chunks[next_id].allocation_type == AllocationType::Free {
            self.merge_free_chunks(chunk_id, next_id)?;
        }

        if !prev_id.is_null() && self.chunks[prev_id].allocation_type == AllocationType::Free {
            self.merge_free_chunks(prev_id, chunk_id)?;
        }
        Ok(())
    }

    fn rename_allocation(
        &mut self,
        chunk_id: Option<std::num::NonZeroU64>,
        name: &str,
    ) -> Result<()> {
        let chunk_id = chunk_id
            .ok_or_else(|| AllocationError::Internal("Chunk ID must be a valid value.".into()))?;

        let chunk_id = MemoryChunkId::from(slotmap::KeyData::from_ffi(chunk_id.get()));

        let chunk = self.chunks.get_mut(chunk_id).ok_or_else(|| {
            AllocationError::Internal(
                "Attempting to rename chunk that is not in chunk list.".into(),
            )
        })?;

        if chunk.allocation_type == AllocationType::Free {
            return Err(AllocationError::Internal(
                "Attempting to rename a freed allocation.".into(),
            ));
        }

        chunk.name = Some(name.into());

        Ok(())
    }

    fn report_memory_leaks(
        &self,
        log_level: Level,
        memory_type_index: usize,
        memory_block_index: usize,
    ) {
        for (chunk_id, chunk) in self.chunks.iter() {
            if chunk.allocation_type == AllocationType::Free {
                continue;
            }
            let empty = "".to_string();
            let name = chunk.name.as_ref().unwrap_or(&empty);
            let backtrace = resolve_backtrace(&chunk.backtrace);

            log!(
                log_level,
                r#"leak detected: {{
    memory type: {}
    memory block: {}
    chunk: {{
        chunk_id: {},
        size: 0x{:x},
        offset: 0x{:x},
        allocation_type: {:?},
        name: {},
        backtrace: {}
    }}
}}"#,
                memory_type_index,
                memory_block_index,
                chunk_id.data().as_ffi(),
                chunk.size,
                chunk.offset,
                chunk.allocation_type,
                name,
                backtrace
            );
        }
    }

    fn report_allocations(&self) -> Vec<AllocationReport> {
        self.chunks
            .iter()
            .filter(|(_key, chunk)| chunk.allocation_type != AllocationType::Free)
            .map(|(_key, chunk)| AllocationReport {
                name: chunk
                    .name
                    .clone()
                    .unwrap_or_else(|| "<Unnamed FreeList allocation>".to_owned()),
                size: chunk.size,
                backtrace: chunk.backtrace.clone(),
            })
            .collect::<Vec<_>>()
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn allocated(&self) -> u64 {
        self.allocated
    }

    fn supports_general_allocations(&self) -> bool {
        true
    }
}
