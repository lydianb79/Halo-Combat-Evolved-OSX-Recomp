#pragma once
#include <stddef.h>
#include <stdint.h>

enum { VFS_NONE, VFS_DISC, VFS_HOST };
typedef struct {
    int kind;
    char host[1024];    /* host path (extracted disc folder or save partition) */
    char rel[512];      /* disc-relative path, '/'-separated */
    int iso_index;      /* >=0 when served from the ISO */
    int is_dir;
    uint32_t size;
} VfsPath;

int vfs_init(void);
int vfs_resolve(const char *xbox_path, VfsPath *out);
void vfs_add_link(const char *name, const char *target);
const char *vfs_query_link(const char *name);
int vfs_have_iso(void);
int vfs_iso_read(int index, uint64_t pos, void *buf, uint32_t len, uint32_t *got);
int vfs_iso_list(const char *dir_rel, int start, char *name, size_t cap, uint32_t *size, int *is_dir);
