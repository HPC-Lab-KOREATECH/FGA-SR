export default {
    "default": {
        file: (target, file, viewer) => `${target.path}/${file}`,
        targetBefore: (target, index, array, viewer) => target,
        targetAfter: (target, index, array, viewer) => target
    }
}
