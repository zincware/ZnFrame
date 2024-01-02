// Replicate znframe.Frame in JavaScript


class Atom {
    constructor({ position, number, arrays }) {
      this.position = position;
      this.number = number;
      this.arrays = arrays;
    }
  }

class Frame {
    constructor(numbers, positions, arrays, info, pbc, cell) {
        this.numbers = numbers;
        this.positions = positions;
        this.arrays = arrays;
        this.info = info;
        this.pbc = pbc;
        this.cell = cell;
    }

    [Symbol.iterator]() {
        let index = 0;
        return {
            next: () => {
                if (index < this.numbers.length) {
                    return { value: new Atom({ position: this.positions[index], number: this.numbers[index], arrays: this.arrays[index] }), done: false };
                } else {
                    return { done: true };
                }
            }
        };
    }
}