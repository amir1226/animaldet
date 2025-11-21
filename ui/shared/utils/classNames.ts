/**
 * Class name mappings for detection models.
 * Matches the CLASS_NAMES mapping from animaldet/app/class_names.py
 */
const CLASS_NAMES: Record<number, string> = {
  1: "Topi",
  2: "Buffalo",
  3: "Kob",
  4: "Warthog",
  5: "Waterbuck",
  6: "Elephant",
}

/**
 * Get class name for a given class ID.
 *
 * @param classId - Class ID (1-indexed)
 * @returns Class name or 'unknown_{classId}'
 */
export function getClassName(classId: number): string {
  return CLASS_NAMES[classId] ?? `unknown_${classId}`
}
