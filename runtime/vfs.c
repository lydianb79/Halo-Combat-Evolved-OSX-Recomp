/* vfs.c — maps Xbox object paths to host storage.
 *
 *   \Device\CdRom0\...                 -> the disc (read directly from the ISO, or an extracted folder)
 *   \Device\Harddisk0\PartitionN\...   -> <save_root>/PartitionN/...
 *   \??\X:\...                         -> resolved through the drive-letter symlink table
 */
#define _GNU_SOURCE
#include "vfs.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "runtime.h"

/* ---------------------------------------------------------------- ISO */
typedef struct IsoEntry {
    char *path;          /* lower-case, '/'-separated, relative to disc root */
    uint64_t offset;     /* absolute byte offset in the image */
    uint32_t size;
    int is_dir;
} IsoEntry;

static FILE *g_iso;
static uint64_t g_iso_base;
static IsoEntry *g_iso_ents;
static int g_iso_count, g_iso_cap;
static pthread_mutex_t g_iso_lock = PTHREAD_MUTEX_INITIALIZER;

static void lower(char *s) { for (; *s; s++) *s = (char)tolower((unsigned char)*s); }

static void iso_add(const char *path, uint64_t off, uint32_t size, int dir) {
    if (g_iso_count == g_iso_cap) {
        g_iso_cap = g_iso_cap ? g_iso_cap * 2 : 256;
        g_iso_ents = realloc(g_iso_ents, sizeof(IsoEntry) * (size_t)g_iso_cap);
    }
    IsoEntry *e = &g_iso_ents[g_iso_count++];
    e->path = strdup(path);
    lower(e->path);
    e->offset = off;
    e->size = size;
    e->is_dir = dir;
}

static void iso_walk(uint32_t sector, uint32_t size, const char *prefix, int depth) {
    if (!size || depth > 16) return;
    uint8_t *d = malloc(size);
    fseeko(g_iso, (off_t)(g_iso_base + (uint64_t)sector * 2048), SEEK_SET);
    if (fread(d, 1, size, g_iso) != size) { free(d); return; }
    uint32_t stack[512], sp = 0;
    uint8_t *seen = calloc(size / 4 + 1, 1);
    stack[sp++] = 0;
    while (sp) {
        uint32_t o = stack[--sp];
        if (o * 4 + 14 > size || seen[o]) continue;
        seen[o] = 1;
        uint8_t *b = d + o * 4;
        uint16_t left = (uint16_t)(b[0] | b[1] << 8), right = (uint16_t)(b[2] | b[3] << 8);
        if (left == 0xFFFF) continue;
        uint32_t start, len;
        memcpy(&start, b + 4, 4);
        memcpy(&len, b + 8, 4);
        uint8_t attr = b[12], nlen = b[13];
        char name[512];
        snprintf(name, sizeof name, "%s%.*s", prefix, nlen, (const char *)b + 14);
        int dir = attr & 0x10;
        iso_add(name, g_iso_base + (uint64_t)start * 2048, len, dir);
        if (dir) {
            char sub[512];
            snprintf(sub, sizeof sub, "%s/", name);
            iso_walk(start, len, sub, depth + 1);
        }
        if (left && sp < 512) stack[sp++] = left;
        if (right && sp < 512) stack[sp++] = right;
    }
    free(seen);
    free(d);
}

static int iso_open(const char *path) {
    g_iso = fopen(path, "rb");
    if (!g_iso) return -1;
    static const uint64_t offs[] = {0, 0x18300000ull, 0xFD90000ull, 0x2080000ull};
    for (int i = 0; i < 4; i++) {
        char magic[20];
        fseeko(g_iso, (off_t)(offs[i] + 32 * 2048), SEEK_SET);
        if (fread(magic, 1, 20, g_iso) == 20 && !memcmp(magic, "MICROSOFT*XBOX*MEDIA", 20)) {
            g_iso_base = offs[i];
            uint32_t root[2];
            if (fread(root, 4, 2, g_iso) != 2) return -1;
            iso_walk(root[0], root[1], "", 0);
            runtime_log("[vfs] ISO %s: %d entries", path, g_iso_count);
            return 0;
        }
    }
    runtime_log("[vfs] %s is not an Xbox disc image", path);
    fclose(g_iso);
    g_iso = NULL;
    return -1;
}

static const IsoEntry *iso_find(const char *rel) {
    char key[512];
    snprintf(key, sizeof key, "%s", rel);
    lower(key);
    if (!*key) return NULL;
    for (int i = 0; i < g_iso_count; i++)
        if (!strcmp(g_iso_ents[i].path, key)) return &g_iso_ents[i];
    return NULL;
}

/* --------------------------------------------------------- symlinks */
typedef struct { char letter_path[64]; char target[256]; } Link;
static Link g_links[32];
static int g_nlinks;

void vfs_add_link(const char *name, const char *target) {
    for (int i = 0; i < g_nlinks; i++)
        if (!strcasecmp(g_links[i].letter_path, name)) {
            snprintf(g_links[i].target, sizeof g_links[i].target, "%s", target);
            return;
        }
    if (g_nlinks < 32) {
        snprintf(g_links[g_nlinks].letter_path, 64, "%s", name);
        snprintf(g_links[g_nlinks].target, 256, "%s", target);
        g_nlinks++;
    }
    runtime_log("[vfs] link %s -> %s", name, target);
}

const char *vfs_query_link(const char *name) {
    for (int i = 0; i < g_nlinks; i++)
        if (!strcasecmp(g_links[i].letter_path, name)) return g_links[i].target;
    return NULL;
}

int vfs_init(void) {
    if (g_cfg.iso_path && *g_cfg.iso_path) iso_open(g_cfg.iso_path);
    vfs_add_link("\\??\\D:", "\\Device\\CdRom0");
    vfs_add_link("\\??\\C:", "\\Device\\Harddisk0\\Partition2");
    vfs_add_link("\\??\\E:", "\\Device\\Harddisk0\\Partition1");
    const char *sr = g_cfg.save_root ? g_cfg.save_root : "xbox_hdd";
    mkdir(sr, 0755);
    char raw[1024];
    snprintf(raw, sizeof raw, "%s/Partition0.raw", sr);
    FILE *f = fopen(raw, "ab");
    if (f) {
        fseek(f, 0, SEEK_END);
        if (ftell(f) < 0x80000) { if (ftruncate(fileno(f), 0x80000)) {} }
        fclose(f);
    }
    return 0;
}

/* --------------------------------------------------------- resolution */
static int path_has_prefix(const char *p, const char *pre, const char **rest) {
    size_t n = strlen(pre);
    if (strncasecmp(p, pre, n)) return 0;
    if (p[n] && p[n] != '\\') return 0;
    *rest = p[n] == '\\' ? p + n + 1 : p + n;
    return 1;
}

/* Case-insensitive lookup of each component under a host directory. */
static void host_join_ci(char *out, size_t cap, const char *root, const char *rel) {
    snprintf(out, cap, "%s", root);
    char comp[256];
    while (*rel) {
        size_t n = strcspn(rel, "\\/");
        snprintf(comp, sizeof comp, "%.*s", (int)n, rel);
        rel += n;
        while (*rel == '\\' || *rel == '/') rel++;
        if (!*comp || !strcmp(comp, ".")) continue;
        char found[256];
        snprintf(found, sizeof found, "%s", comp);
        DIR *d = opendir(out);
        if (d) {
            struct dirent *de;
            while ((de = readdir(d)))
                if (!strcasecmp(de->d_name, comp)) { snprintf(found, sizeof found, "%s", de->d_name); break; }
            closedir(d);
        }
        size_t l = strlen(out);
        snprintf(out + l, cap - l, "/%s", found);
    }
}

int vfs_resolve(const char *xbox_path, VfsPath *out) {
    memset(out, 0, sizeof *out);
    char p[512];
    snprintf(p, sizeof p, "%s", xbox_path);
    /* expand \??\X: and bare X: */
    for (int iter = 0; iter < 4; iter++) {
        const char *rest;
        char drive[16];
        if (isalpha((unsigned char)p[0]) && p[1] == ':') {
            char t[512];
            snprintf(t, sizeof t, "\\??\\%s", p);
            snprintf(p, sizeof p, "%s", t);
        }
        if (!strncmp(p, "\\??\\", 4) && p[5] == ':') {
            snprintf(drive, sizeof drive, "\\??\\%c:", toupper((unsigned char)p[4]));
            const char *tgt = vfs_query_link(drive);
            if (!tgt) return -1;
            rest = p + 6;
            char t[512];
            snprintf(t, sizeof t, "%s%s", tgt, rest);
            snprintf(p, sizeof p, "%s", t);
            continue;
        }
        break;
    }
    const char *rest;
    if (path_has_prefix(p, "\\Device\\CdRom0", &rest)) {
        out->kind = VFS_DISC;
        snprintf(out->rel, sizeof out->rel, "%s", rest);
        for (char *c = out->rel; *c; c++) if (*c == '\\') *c = '/';
        if (g_iso) {
            out->iso_index = -1;
            if (!*out->rel) { out->is_dir = 1; return 0; }
            const IsoEntry *e = iso_find(out->rel);
            if (!e) return -1;
            out->iso_index = (int)(e - g_iso_ents);
            out->is_dir = e->is_dir;
            out->size = e->size;
            return 0;
        }
        host_join_ci(out->host, sizeof out->host, g_cfg.game_root ? g_cfg.game_root : ".", rest);
        return 0;
    }
    if (path_has_prefix(p, "\\Device\\Harddisk0\\Partition0", &rest) && !*rest) {
        /* raw config area of the hard disk (refurb info, title persist data) */
        out->kind = VFS_HOST;
        snprintf(out->host, sizeof out->host, "%s/Partition0.raw", g_cfg.save_root ? g_cfg.save_root : "xbox_hdd");
        return 0;
    }
    for (int part = 0; part <= 7; part++) {
        char pre[64];
        snprintf(pre, sizeof pre, "\\Device\\Harddisk0\\Partition%d", part);
        if (path_has_prefix(p, pre, &rest)) {
            out->kind = VFS_HOST;
            char root[512];
            snprintf(root, sizeof root, "%s/Partition%d", g_cfg.save_root ? g_cfg.save_root : "xbox_hdd", part);
            mkdir(root, 0755);
            host_join_ci(out->host, sizeof out->host, root, rest);
            return 0;
        }
    }
    return -1;
}

int vfs_iso_read(int index, uint64_t pos, void *buf, uint32_t len, uint32_t *got) {
    const IsoEntry *e = &g_iso_ents[index];
    *got = 0;
    if (pos >= e->size) return 0;
    if (pos + len > e->size) len = (uint32_t)(e->size - pos);
    pthread_mutex_lock(&g_iso_lock);
    fseeko(g_iso, (off_t)(e->offset + pos), SEEK_SET);
    *got = (uint32_t)fread(buf, 1, len, g_iso);
    pthread_mutex_unlock(&g_iso_lock);
    return 0;
}

int vfs_iso_list(const char *dir_rel, int start, char *name, size_t cap, uint32_t *size, int *is_dir) {
    char key[512];
    snprintf(key, sizeof key, "%s", dir_rel);
    lower(key);
    size_t kl = strlen(key);
    int n = 0;
    for (int i = 0; i < g_iso_count; i++) {
        const char *p = g_iso_ents[i].path;
        if (kl && (strncmp(p, key, kl) || p[kl] != '/')) continue;
        const char *leaf = kl ? p + kl + 1 : p;
        if (strchr(leaf, '/')) continue;
        if (n++ == start) {
            snprintf(name, cap, "%s", leaf);
            *size = g_iso_ents[i].size;
            *is_dir = g_iso_ents[i].is_dir;
            return 1;
        }
    }
    return 0;
}

int vfs_have_iso(void) { return g_iso != NULL; }
